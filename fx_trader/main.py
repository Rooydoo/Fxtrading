#!/usr/bin/env python3
"""
FX Machine Learning Trading System
メインエントリーポイント
"""
import argparse
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fx_trader.log"),
    ],
)
logger = logging.getLogger(__name__)

from src.core.mode import ModeManager, EnvironmentManager, SystemState, TradingMode
from src.core.scheduler import TradingScheduler, MarketHoursChecker, HealthChecker
from src.data.gmo_client import GMOForexClient
from src.data.fetcher import DataFetcher, PaperDataFetcher
from src.data.cache import OHLCVCache, CachedDataFetcher
from src.data.economic_calendar import EconomicCalendar, TradingFilter, CalendarUpdater
from src.features.builder import FeatureBuilder
from src.features.selector import FeatureSelector
from src.model.predictor import SignalPredictor, PredictionLogger
from src.trading.position import PositionManager, TradeHistory, Side
from src.trading.risk_manager import RiskManager
from src.trading.executor import (
    TradeExecutor,
    LiveOrderExecutor,
    PaperOrderExecutor,
)
from src.trading.trailing_stop import TrailingStopManager, TrailingStopConfig, TrailingMethod
from src.trading.position_recovery import PositionRecoveryManager, PositionSynchronizer, RecoveryHandler
from src.trading.partial_close import PartialCloseManager, load_partial_close_config
from src.notification.telegram import TelegramNotifier
from src.notification.reporter import PerformanceReporter, ReportScheduler
from src.notification.bot_commands import TradingBotCommands
from src.monitoring.performance_tracker import PerformanceTracker
from src.trading.paper_simulator import PaperTradingSimulator


class FXTradingSystem:
    """FXトレーディングシステム"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path

        # ディレクトリ作成
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("data/backups").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        # モード管理
        self.mode_manager = ModeManager(config_path)
        self.env_manager = EnvironmentManager(self.mode_manager.mode)
        self.system_state = SystemState()

        # 環境変数検証
        env_result = self.env_manager.validate()
        if not env_result["valid"]:
            logger.error("Environment validation failed")
            raise RuntimeError(f"Missing required environment variables: {env_result['required']['missing']}")

        # コンポーネント初期化
        self._init_components()

        # ポジション復旧
        self._perform_recovery()

        logger.info(f"FX Trading System initialized in {self.mode_manager.mode.value} mode")

    def _init_components(self) -> None:
        """コンポーネントを初期化"""
        credentials = self.env_manager.get_credentials()

        # APIクライアント
        self.client = GMOForexClient(
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
        )

        # データキャッシュ
        self.ohlcv_cache = OHLCVCache(
            db_path="data/ohlcv_cache.db",
            max_age_hours=24,
        )

        # データフェッチャー（キャッシュ付き）
        if self.mode_manager.is_paper():
            base_fetcher = PaperDataFetcher(self.client)
        else:
            base_fetcher = DataFetcher(self.client)

        self.fetcher = CachedDataFetcher(base_fetcher, self.ohlcv_cache)

        # 経済指標カレンダー
        self.economic_calendar = EconomicCalendar(
            calendar_file="data/economic_calendar.json",
        )
        self.trading_filter = TradingFilter(self.economic_calendar)

        # サンプルカレンダー作成（初回のみ）
        if not Path("data/economic_calendar.json").exists():
            updater = CalendarUpdater(self.economic_calendar)
            updater.create_sample_calendar()

        # 特徴量
        self.feature_builder = FeatureBuilder("config/features.yaml")
        self.feature_selector = FeatureSelector()

        # モデル
        self.predictor: Optional[SignalPredictor] = None
        self.prediction_logger = PredictionLogger()

        # トレーディング
        self.position_manager = PositionManager(max_positions=3)
        self.risk_manager = RiskManager("config/risk_params.yaml")
        self.trade_history = TradeHistory("data/trades.db")

        # トレーリングストップ
        trailing_config = self._load_trailing_config()
        self.trailing_stop_manager = TrailingStopManager(trailing_config)

        # 部分利確
        partial_close_config = load_partial_close_config("config/risk_params.yaml")
        self.partial_close_manager = PartialCloseManager(partial_close_config)

        # ポジション復旧マネージャー
        self.recovery_manager = PositionRecoveryManager(
            state_file="data/position_state.json",
            backup_dir="data/backups",
        )

        # 本番モードの場合はAPI同期も設定
        if self.mode_manager.is_live():
            self.position_synchronizer = PositionSynchronizer(self.client)
        else:
            self.position_synchronizer = None

        self.recovery_handler = RecoveryHandler(
            self.recovery_manager,
            self.position_synchronizer,
        )

        # ペーパートレードシミュレーター
        self.paper_simulator: Optional[PaperTradingSimulator] = None
        if self.mode_manager.is_paper():
            initial_balance = self.mode_manager.get_config("paper_trading.initial_balance", 1000000)
            self.paper_simulator = PaperTradingSimulator(
                initial_balance=initial_balance,
                data_dir="data/paper_trading",
            )

        # 注文執行
        if self.mode_manager.is_live():
            executor = LiveOrderExecutor(self.client)
        else:
            initial_balance = self.mode_manager.get_config("paper_trading.initial_balance", 1000000)
            executor = PaperOrderExecutor(self.client, initial_balance)

        self.trade_executor = TradeExecutor(
            executor=executor,
            position_manager=self.position_manager,
            risk_manager=self.risk_manager,
            trade_history=self.trade_history,
        )

        # 通知
        self.notifier = TelegramNotifier(
            bot_token=credentials["telegram_token"],
            chat_id=credentials["telegram_chat_id"],
        )

        # Telegramボットコマンド
        self.bot_commands = TradingBotCommands(
            bot_token=credentials["telegram_token"],
            chat_id=credentials["telegram_chat_id"],
            trading_system=self,
        )

        # レポーター
        self.reporter = PerformanceReporter(self.trade_history, self.notifier)
        self.report_scheduler = ReportScheduler(self.reporter)

        # モニタリング
        self.performance_tracker = PerformanceTracker()

        # スケジューラー
        self.scheduler = TradingScheduler(interval_minutes=15)
        self.market_checker = MarketHoursChecker()
        self.health_checker = HealthChecker()

    def _load_trailing_config(self) -> TrailingStopConfig:
        """トレーリングストップ設定を読み込み"""
        try:
            import yaml
            with open("config/risk_params.yaml", "r") as f:
                config = yaml.safe_load(f)

            ts_config = config.get("trailing_stop", {})

            if not ts_config.get("enabled", True):
                return TrailingStopConfig(enabled=False)

            method_map = {
                "fixed_pips": TrailingMethod.FIXED_PIPS,
                "atr_based": TrailingMethod.ATR_BASED,
                "percent": TrailingMethod.PERCENT,
                "breakeven": TrailingMethod.BREAKEVEN,
                "step": TrailingMethod.STEP,
            }

            return TrailingStopConfig(
                enabled=True,
                method=method_map.get(ts_config.get("method", "atr_based"), TrailingMethod.ATR_BASED),
                trail_pips=ts_config.get("trail_pips", 30.0),
                atr_multiplier=ts_config.get("atr_multiplier", 1.5),
                trail_percent=ts_config.get("trail_percent", 0.005),
                breakeven_trigger_pips=ts_config.get("breakeven_trigger_pips", 20.0),
                breakeven_offset_pips=ts_config.get("breakeven_offset_pips", 5.0),
                step_pips=ts_config.get("step_pips", 20.0),
                activation_pips=ts_config.get("activation_pips", 10.0),
                min_trail_distance=ts_config.get("min_trail_distance", 15.0),
            )
        except Exception as e:
            logger.warning(f"Failed to load trailing config: {e}, using defaults")
            return TrailingStopConfig()

    def _perform_recovery(self) -> None:
        """起動時のポジション復旧"""
        if self.recovery_manager.needs_recovery():
            logger.info("Performing position recovery...")
            result = self.recovery_handler.perform_recovery(
                position_manager=self.position_manager,
                trailing_stop_manager=self.trailing_stop_manager,
                is_live=self.mode_manager.is_live(),
            )

            if result["positions_recovered"] > 0:
                logger.info(f"Recovered {result['positions_recovered']} positions")
                self.notifier.send_message(
                    f"🔄 ポジション復旧完了\n"
                    f"復元: {result['positions_recovered']}件\n"
                    f"トレーリング: {result['trailing_states_recovered']}件"
                )

            if result["warnings"]:
                for warning in result["warnings"]:
                    logger.warning(warning)

    def _save_state(self) -> None:
        """状態を保存"""
        system_state = {
            "mode": self.mode_manager.mode.value,
            "timestamp": datetime.now().isoformat(),
            "daily_pnl": self.risk_manager.daily_pnl,
            "consecutive_losses": self.risk_manager.consecutive_losses,
        }

        self.recovery_handler.save_current_state(
            position_manager=self.position_manager,
            system_state=system_state,
            trailing_stop_manager=self.trailing_stop_manager,
        )

    def load_model(self, model_path: str) -> None:
        """
        モデルをロード

        Args:
            model_path: モデルファイルパス
        """
        self.predictor = SignalPredictor(model_path)
        logger.info(f"Model loaded from {model_path}")

    def trading_cycle(self) -> None:
        """トレーディングサイクル (15分ごとに実行)"""
        try:
            self.health_checker.heartbeat()

            # 市場オープンチェック
            if not self.market_checker.is_market_open():
                logger.debug("Market is closed")
                return

            # システム状態チェック
            if not self.system_state.can_trade():
                logger.debug(f"Trading disabled: {self.system_state.state}")
                return

            # 各通貨ペアを処理
            symbols = self.mode_manager.get_config("trading.currency_pairs", ["EUR_USD", "USD_JPY"])

            for symbol in symbols:
                self._process_symbol(symbol)

            # ポジションチェック（トレーリングストップ含む）
            self._check_positions()

            # 状態保存（定期）
            self._save_state()

            # レポートスケジュール
            balance = self._get_balance()
            positions = [p.to_dict() for p in self.position_manager.get_open_positions()]
            self.report_scheduler.check_and_send_reports(positions, balance)

        except Exception as e:
            logger.exception(f"Trading cycle error: {e}")
            self.health_checker.record_error(e, "trading_cycle")
            self.notifier.send_emergency_alert("システムエラー", str(e))

    def _process_symbol(self, symbol: str) -> None:
        """通貨ペアを処理"""
        try:
            # 経済指標チェック
            can_trade_calendar, calendar_reason = self.trading_filter.can_trade(symbol)
            if not can_trade_calendar:
                logger.info(f"Trading blocked for {symbol}: {calendar_reason}")
                return

            # データ取得（キャッシュ使用）
            df = self.fetcher.fetch_ohlcv(symbol, interval="15m", days=7)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return

            # 上位時間軸データ
            df_1h = self.fetcher.fetch_ohlcv(symbol, interval="1h", days=7)

            # 特徴量生成
            df_features = self.feature_builder.build_all_features(df, df_1h)

            # モデル予測
            if self.predictor is None:
                logger.warning("No model loaded")
                return

            signal, confidence, details = self.predictor.generate_signal(df_features)

            # 予測ログ
            self.prediction_logger.log_prediction(symbol, signal, confidence, details)

            # シグナル処理
            if signal == 0:
                return

            # 取引可能チェック
            balance = self._get_balance()
            can_trade, reason = self.risk_manager.can_trade(balance)
            if not can_trade:
                logger.info(f"Cannot trade {symbol}: {reason}")
                return

            # スプレッドチェック
            spread, is_normal = self.fetcher.fetcher.calculate_spread(symbol)
            is_spread_ok, _ = self.risk_manager.check_spread(symbol, spread)
            if not is_spread_ok:
                logger.info(f"Spread too wide for {symbol}: {spread} pips")
                return

            # ATR取得
            atr = df_features["atr_14"].iloc[-1]

            # エントリー
            side = Side.LONG if signal == 1 else Side.SHORT
            position = self.trade_executor.open_trade(
                symbol=symbol,
                side=side,
                confidence=confidence,
                atr=atr,
                balance=balance,
            )

            if position:
                # トレーリングストップに登録
                self.trailing_stop_manager.register_position(
                    position_id=position.id,
                    symbol=symbol,
                    side="long" if side == Side.LONG else "short",
                    entry_price=position.entry_price,
                    stop_loss=position.stop_loss,
                )

                # 部分利確に登録
                self.partial_close_manager.register_position(
                    position_id=position.id,
                    symbol=symbol,
                    side="long" if side == Side.LONG else "short",
                    entry_price=position.entry_price,
                    size=position.size,
                )

                # 通知
                max_loss, max_loss_pct = self.risk_manager.calculate_max_loss(
                    balance, position.size, position.entry_price, position.stop_loss, symbol
                )
                self.notifier.send_entry_notification(
                    symbol=symbol,
                    side=side.value,
                    entry_price=position.entry_price,
                    size=position.size,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    confidence=confidence,
                    max_loss_amount=max_loss,
                    max_loss_percent=max_loss_pct,
                )

                # 状態保存
                self._save_state()

        except Exception as e:
            logger.exception(f"Error processing {symbol}: {e}")

    def _check_positions(self) -> None:
        """ポジションをチェック（トレーリングストップ含む）"""
        try:
            positions = self.position_manager.get_open_positions()
            if not positions:
                return

            # 現在価格とATRを取得
            prices = {}
            atrs = {}
            for pos in positions:
                try:
                    ticker = self.fetcher.fetcher.fetch_ticker(pos.symbol)
                    prices[pos.symbol] = ticker["bid"]

                    # ATR取得（トレーリングストップ用）
                    df = self.fetcher.fetch_ohlcv(pos.symbol, interval="15m", days=2)
                    if not df.empty and "atr_14" not in df.columns:
                        df_features = self.feature_builder.build_all_features(df)
                        if "atr_14" in df_features.columns:
                            atrs[pos.symbol] = df_features["atr_14"].iloc[-1]
                except Exception:
                    prices[pos.symbol] = pos.entry_price

            # トレーリングストップ更新 & 部分利確チェック
            for pos in positions:
                current_price = prices.get(pos.symbol)
                atr = atrs.get(pos.symbol)

                if current_price:
                    # トレーリングストップ更新
                    updated, new_sl = self.trailing_stop_manager.update(
                        position_id=pos.id,
                        current_price=current_price,
                        atr=atr,
                    )

                    if updated and new_sl:
                        # PositionManagerのSLを更新
                        self.position_manager.update_stop_loss(pos.id, new_sl)
                        logger.info(f"Trailing stop updated: {pos.id}, new SL={new_sl:.5f}")

                    # 部分利確チェック
                    partial_closes = self.partial_close_manager.check_and_close(
                        position_id=pos.id,
                        current_price=current_price,
                    )

                    for pc in partial_closes:
                        # 部分決済を実行（実際の注文はexecutorで行う）
                        close_size = pc["size"]
                        logger.info(
                            f"Partial close triggered: {pos.id}, "
                            f"size={close_size}, trigger={pc['trigger_pips']}pips"
                        )

                        # 部分決済の記録
                        # 実際のPnLは決済後に計算
                        estimated_pnl = 0  # TODO: 実際の決済処理と連携
                        self.partial_close_manager.record_partial_close(
                            position_id=pos.id,
                            level_index=pc["level_index"],
                            closed_size=close_size,
                            close_price=current_price,
                            pnl=estimated_pnl,
                        )

                        # SLをエントリー価格に移動（設定されている場合）
                        if pc.get("move_sl_to_entry"):
                            self.position_manager.update_stop_loss(pos.id, pos.entry_price)
                            logger.info(f"SL moved to entry: {pos.id}, SL={pos.entry_price:.5f}")

            # SL/TPチェック
            closed = self.trade_executor.check_and_close_positions(prices)

            for close_info in closed:
                pos_id = close_info["position_id"]
                pnl = close_info.get("pnl", 0)

                # トレーリングストップから登録解除
                self.trailing_stop_manager.unregister_position(pos_id)

                # 部分利確から登録解除
                self.partial_close_manager.unregister_position(pos_id)

                # 通知
                logger.info(f"Position closed: {pos_id}, PnL: {pnl:.2f}")

                # 状態保存
                self._save_state()

        except Exception as e:
            logger.exception(f"Position check error: {e}")

    def _get_balance(self) -> float:
        """残高を取得"""
        if self.mode_manager.is_paper():
            return self.trade_executor.executor.get_balance()
        else:
            try:
                response = self.client.get_account_margin()
                if response.get("status") == 0:
                    return float(response["data"]["availableAmount"])
            except Exception as e:
                logger.error(f"Failed to get balance: {e}")

            return self.mode_manager.get_config("paper_trading.initial_balance", 1000000)

    def start(self) -> None:
        """システムを開始"""
        logger.info("Starting FX Trading System")

        self.system_state.set_running()

        # シグナルハンドラー設定
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Telegramボットコマンド開始
        self.bot_commands.start_polling()

        # 今後の経済指標イベント
        upcoming_events = self.trading_filter.get_blocked_periods("EUR_USD", hours=24)

        # Telegram通知
        mode_str = self.mode_manager.mode.value.upper()
        balance = self._get_balance()
        positions_count = len(self.position_manager.get_open_positions())

        startup_msg = (
            f"🚀 FX Trading System 起動\n"
            f"モード: {mode_str}\n"
            f"残高: ¥{balance:,.0f}\n"
            f"オープンポジション: {positions_count}件\n"
            f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        if upcoming_events:
            startup_msg += f"\n⚠️ 24h以内の重要指標: {len(upcoming_events)}件"

        startup_msg += "\n\n/help でコマンド一覧を表示"

        self.notifier.send_message(startup_msg)

        # コールバック登録
        self.scheduler.add_callback(self.trading_cycle)
        self.scheduler.add_error_handler(lambda e: self.health_checker.record_error(e, "scheduler"))

        # スケジューラー開始
        self.scheduler.start()

    def _signal_handler(self, signum, frame) -> None:
        """シグナルハンドラー"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def stop(self) -> None:
        """システムを停止"""
        logger.info("Stopping FX Trading System")

        self.system_state.request_shutdown()
        self.scheduler.stop()
        self.bot_commands.stop_polling()

        # 状態保存
        self._save_state()

        # ペーパーシミュレーター状態保存
        if self.paper_simulator:
            self.paper_simulator.save_state()
            self.paper_simulator.record_daily()

        # 通知
        positions_count = len(self.position_manager.get_open_positions())
        self.notifier.send_message(
            f"🛑 FX Trading System 停止\n"
            f"オープンポジション: {positions_count}件（保存済み）\n"
            f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="FX ML Trading System")
    parser.add_argument("--config", default="config/settings.yaml", help="設定ファイルパス")
    parser.add_argument("--model", help="モデルファイルパス")
    parser.add_argument("--mode", choices=["live", "paper"], help="動作モード")
    parser.add_argument("--once", action="store_true", help="1回だけ実行")
    args = parser.parse_args()

    # ワーキングディレクトリ
    os.chdir(Path(__file__).parent)

    try:
        system = FXTradingSystem(args.config)

        if args.model:
            system.load_model(args.model)
        else:
            # デフォルトモデルを探す
            default_model = Path("models/lightgbm_model.pkl")
            if default_model.exists():
                system.load_model(str(default_model))
            else:
                logger.warning("No model specified. System will run without predictions.")

        if args.once:
            system.trading_cycle()
        else:
            system.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

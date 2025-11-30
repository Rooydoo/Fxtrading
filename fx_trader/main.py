#!/usr/bin/env python3
"""
FX Machine Learning Trading System
メインエントリーポイント
"""
import argparse
import logging
import os
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

        logger.info(f"FX Trading System initialized in {self.mode_manager.mode.value} mode")

    def _init_components(self) -> None:
        """コンポーネントを初期化"""
        credentials = self.env_manager.get_credentials()

        # APIクライアント
        self.client = GMOForexClient(
            api_key=credentials["api_key"],
            api_secret=credentials["api_secret"],
        )

        # データフェッチャー
        if self.mode_manager.is_paper():
            self.fetcher = PaperDataFetcher(self.client)
        else:
            self.fetcher = DataFetcher(self.client)

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

            # SL/TPチェック
            self._check_positions()

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
            # データ取得
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
            spread, is_normal = self.fetcher.calculate_spread(symbol)
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

        except Exception as e:
            logger.exception(f"Error processing {symbol}: {e}")

    def _check_positions(self) -> None:
        """ポジションをチェック"""
        try:
            positions = self.position_manager.get_open_positions()
            if not positions:
                return

            # 現在価格取得
            prices = {}
            for pos in positions:
                try:
                    ticker = self.fetcher.fetch_ticker(pos.symbol)
                    prices[pos.symbol] = ticker["bid"]
                except Exception:
                    prices[pos.symbol] = pos.entry_price

            # SL/TPチェック
            closed = self.trade_executor.check_and_close_positions(prices)

            for close_info in closed:
                pos_id = close_info["position_id"]
                # 通知 (実際の実装ではポジション情報を取得して通知)
                logger.info(f"Position closed: {pos_id}, PnL: {close_info['pnl']:.2f}")

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

        # Telegramボットコマンド開始
        self.bot_commands.start_polling()

        # Telegram通知
        mode_str = self.mode_manager.mode.value.upper()
        balance = self._get_balance()
        self.notifier.send_message(
            f"🚀 FX Trading System 起動\n"
            f"モード: {mode_str}\n"
            f"残高: ¥{balance:,.0f}\n"
            f"時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"/help でコマンド一覧を表示"
        )

        # コールバック登録
        self.scheduler.add_callback(self.trading_cycle)
        self.scheduler.add_error_handler(lambda e: self.health_checker.record_error(e, "scheduler"))

        # スケジューラー開始
        self.scheduler.start()

    def stop(self) -> None:
        """システムを停止"""
        logger.info("Stopping FX Trading System")

        self.system_state.request_shutdown()
        self.scheduler.stop()
        self.bot_commands.stop_polling()

        # ペーパーシミュレーター状態保存
        if self.paper_simulator:
            self.paper_simulator.save_state()
            self.paper_simulator.record_daily()

        # 通知
        self.notifier.send_message(
            f"🛑 FX Trading System 停止\n"
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

"""
リスク管理モジュール
ポジションサイジング、損失制限、リスク計算
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .position import Position, PositionManager, Side, TradeHistory

logger = logging.getLogger(__name__)


class RiskManager:
    """リスク管理クラス"""

    def __init__(self, config_path: str = "config/risk_params.yaml"):
        """
        Args:
            config_path: リスクパラメータ設定ファイルのパス
        """
        self.config = self._load_config(config_path)
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        self.last_trade_result: Optional[bool] = None
        self.trading_halted = False
        self.halt_reason: Optional[str] = None

    def _load_config(self, path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "position_risk": {
                "long": {
                    "risk_per_trade": 0.01,
                    "stop_loss": {"method": "atr", "atr_multiplier": 1.5},
                    "take_profit": {"method": "atr", "atr_multiplier": 2.0},
                },
                "short": {
                    "risk_per_trade": 0.01,
                    "stop_loss": {"method": "atr", "atr_multiplier": 1.5},
                    "take_profit": {"method": "atr", "atr_multiplier": 2.5},
                },
            },
            "capital_management": {
                "daily_loss_limit": {"enabled": True, "percent": 0.02},
                "weekly_drawdown_limit": {"enabled": True, "percent": 0.05},
                "consecutive_loss": {"enabled": True, "threshold": 5},
            },
        }

    def calculate_position_size(
        self,
        balance: float,
        side: Side,
        entry_price: float,
        stop_loss_price: float,
        symbol: str,
    ) -> float:
        """
        ポジションサイズを計算

        Args:
            balance: 口座残高
            side: 売買方向
            entry_price: エントリー価格
            stop_loss_price: ストップロス価格
            symbol: 通貨ペア

        Returns:
            ポジションサイズ (通貨単位)
        """
        # ロング/ショート別のリスク設定を取得
        side_key = "long" if side == Side.LONG else "short"
        risk_config = self.config.get("position_risk", {}).get(side_key, {})
        risk_per_trade = risk_config.get("risk_per_trade", 0.01)

        # 許容損失額
        risk_amount = balance * risk_per_trade

        # ストップまでの距離 (絶対値)
        sl_distance = abs(entry_price - stop_loss_price)

        if sl_distance == 0:
            logger.warning("Stop loss distance is zero, using minimum size")
            return 1000  # 最小ロット

        # pip値の計算
        if "JPY" in symbol:
            pip_value = 100  # 1 pip = 0.01
        else:
            pip_value = 10000  # 1 pip = 0.0001

        sl_pips = sl_distance * pip_value

        # ポジションサイズ = 許容損失 / (SL pips × 1pipの価値)
        # 1万通貨あたりの1pipの価値を使用
        if "JPY" in symbol:
            pip_value_per_lot = 100  # 1万通貨で100円/pip
        else:
            pip_value_per_lot = 1  # 簡略化 (実際はUSD換算が必要)

        position_size = risk_amount / (sl_pips * pip_value_per_lot / pip_value)

        # 1000通貨単位に丸め
        position_size = max(1000, round(position_size / 1000) * 1000)

        logger.debug(f"Position size calculated: {position_size} (risk={risk_amount}, sl_pips={sl_pips})")

        return position_size

    def calculate_stop_loss(
        self,
        side: Side,
        entry_price: float,
        atr: float,
        symbol: str,
    ) -> float:
        """
        ストップロス価格を計算

        Args:
            side: 売買方向
            entry_price: エントリー価格
            atr: ATR値
            symbol: 通貨ペア

        Returns:
            ストップロス価格
        """
        side_key = "long" if side == Side.LONG else "short"
        sl_config = self.config.get("position_risk", {}).get(side_key, {}).get("stop_loss", {})

        method = sl_config.get("method", "atr")
        min_pips = sl_config.get("min_pips", 20)
        max_pips = sl_config.get("max_pips", 40)

        if method == "atr":
            multiplier = sl_config.get("atr_multiplier", 1.5)
            sl_distance = atr * multiplier
        else:
            # 固定pip
            sl_distance = sl_config.get("fixed_pips", 30)
            if "JPY" in symbol:
                sl_distance /= 100
            else:
                sl_distance /= 10000

        # pip単位に変換して範囲制限
        if "JPY" in symbol:
            sl_pips = sl_distance * 100
            sl_pips = max(min_pips, min(max_pips, sl_pips))
            sl_distance = sl_pips / 100
        else:
            sl_pips = sl_distance * 10000
            sl_pips = max(min_pips, min(max_pips, sl_pips))
            sl_distance = sl_pips / 10000

        if side == Side.LONG:
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

        return stop_loss

    def calculate_take_profit(
        self,
        side: Side,
        entry_price: float,
        atr: float,
        symbol: str,
    ) -> float:
        """
        テイクプロフィット価格を計算

        Args:
            side: 売買方向
            entry_price: エントリー価格
            atr: ATR値
            symbol: 通貨ペア

        Returns:
            テイクプロフィット価格
        """
        side_key = "long" if side == Side.LONG else "short"
        tp_config = self.config.get("position_risk", {}).get(side_key, {}).get("take_profit", {})

        method = tp_config.get("method", "atr")
        min_pips = tp_config.get("min_pips", 30)
        max_pips = tp_config.get("max_pips", 60)

        if method == "atr":
            multiplier = tp_config.get("atr_multiplier", 2.0)
            tp_distance = atr * multiplier
        else:
            tp_distance = tp_config.get("fixed_pips", 40)
            if "JPY" in symbol:
                tp_distance /= 100
            else:
                tp_distance /= 10000

        # pip単位に変換して範囲制限
        if "JPY" in symbol:
            tp_pips = tp_distance * 100
            tp_pips = max(min_pips, min(max_pips, tp_pips))
            tp_distance = tp_pips / 100
        else:
            tp_pips = tp_distance * 10000
            tp_pips = max(min_pips, min(max_pips, tp_pips))
            tp_distance = tp_pips / 10000

        if side == Side.LONG:
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        return take_profit

    def calculate_max_loss(
        self,
        balance: float,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        symbol: str,
    ) -> Tuple[float, float]:
        """
        最大損失を計算

        Args:
            balance: 口座残高
            position_size: ポジションサイズ
            entry_price: エントリー価格
            stop_loss: ストップロス価格
            symbol: 通貨ペア

        Returns:
            (最大損失額, 最大損失率)
        """
        sl_distance = abs(entry_price - stop_loss)

        # 損失計算
        max_loss_amount = sl_distance * position_size

        # JPYペアの場合は円換算
        if "JPY" not in symbol:
            # USD建ての場合、概算で円換算 (実際は為替レートが必要)
            max_loss_amount *= 150  # 仮のレート

        max_loss_percent = max_loss_amount / balance

        return max_loss_amount, max_loss_percent

    def can_trade(self, balance: float) -> Tuple[bool, Optional[str]]:
        """
        取引可能か判定

        Args:
            balance: 現在の口座残高

        Returns:
            (取引可能フラグ, 理由)
        """
        if self.trading_halted:
            return False, self.halt_reason

        capital_config = self.config.get("capital_management", {})

        # デイリー損失制限
        daily_limit = capital_config.get("daily_loss_limit", {})
        if daily_limit.get("enabled", True):
            limit_percent = daily_limit.get("percent", 0.02)
            if self.daily_pnl < -balance * limit_percent:
                return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # 連敗制限
        consec_config = capital_config.get("consecutive_loss", {})
        if consec_config.get("enabled", True):
            threshold = consec_config.get("threshold", 5)
            if self.consecutive_losses >= threshold:
                return False, f"Consecutive losses reached: {self.consecutive_losses}"

        return True, None

    def update_trade_result(self, pnl: float) -> None:
        """
        取引結果を更新

        Args:
            pnl: 取引損益
        """
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        # 連敗カウント
        if pnl < 0:
            self.consecutive_losses += 1
            self.last_trade_result = False
        else:
            self.consecutive_losses = 0
            self.last_trade_result = True

        logger.debug(f"Trade result updated: PnL={pnl:.2f}, Daily={self.daily_pnl:.2f}, Consecutive losses={self.consecutive_losses}")

    def reset_daily(self) -> None:
        """デイリー統計をリセット"""
        self.daily_pnl = 0.0
        logger.info("Daily PnL reset")

    def reset_weekly(self) -> None:
        """週次統計をリセット"""
        self.weekly_pnl = 0.0
        self.consecutive_losses = 0
        logger.info("Weekly stats reset")

    def halt_trading(self, reason: str) -> None:
        """取引を停止"""
        self.trading_halted = True
        self.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")

    def resume_trading(self) -> None:
        """取引を再開"""
        self.trading_halted = False
        self.halt_reason = None
        logger.info("Trading resumed")

    def check_spread(
        self,
        symbol: str,
        current_spread: float,
    ) -> Tuple[bool, float]:
        """
        スプレッドをチェック

        Args:
            symbol: 通貨ペア
            current_spread: 現在のスプレッド (pips)

        Returns:
            (正常フラグ, 通常スプレッド比率)
        """
        spread_config = self.config.get("trading_conditions", {}).get("spread_check", {})

        if not spread_config.get("enabled", True):
            return True, 1.0

        normal_spreads = spread_config.get("normal_spread", {})
        normal = normal_spreads.get(symbol, 2.0)
        max_multiplier = spread_config.get("max_spread_multiplier", 2.0)

        ratio = current_spread / normal
        is_normal = ratio <= max_multiplier

        if not is_normal:
            logger.warning(f"Abnormal spread detected for {symbol}: {current_spread} pips (normal={normal})")

        return is_normal, ratio

    def get_risk_summary(
        self,
        balance: float,
        positions: List[Position],
        prices: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        リスクサマリーを取得

        Args:
            balance: 口座残高
            positions: オープンポジションリスト
            prices: 現在価格辞書

        Returns:
            リスクサマリー
        """
        unrealized_pnl = sum(
            p.calculate_pnl(prices.get(p.symbol, p.entry_price))
            for p in positions
        )

        total_exposure = sum(p.size * p.entry_price for p in positions)

        max_potential_loss = sum(p.max_loss_amount for p in positions)

        return {
            "balance": balance,
            "unrealized_pnl": unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(positions),
            "total_exposure": total_exposure,
            "exposure_percent": total_exposure / balance if balance > 0 else 0,
            "max_potential_loss": max_potential_loss,
            "max_potential_loss_percent": max_potential_loss / balance if balance > 0 else 0,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }


class VolatilityAdjustedSizer:
    """ボラティリティ調整ポジションサイザー"""

    def __init__(
        self,
        target_volatility: float = 0.01,
        atr_period: int = 14,
    ):
        """
        Args:
            target_volatility: 目標日次ボラティリティ
            atr_period: ATR期間
        """
        self.target_volatility = target_volatility
        self.atr_period = atr_period

    def calculate_size(
        self,
        balance: float,
        atr: float,
        price: float,
        max_size: Optional[float] = None,
    ) -> float:
        """
        ボラティリティ調整済みサイズを計算

        Args:
            balance: 口座残高
            atr: ATR値
            price: 現在価格
            max_size: 最大サイズ

        Returns:
            調整済みポジションサイズ
        """
        # 日次ボラティリティ (ATRベース)
        daily_vol = atr / price

        if daily_vol == 0:
            logger.warning("Daily volatility is zero")
            return 1000

        # 目標ボラティリティに対するスケール
        vol_scale = self.target_volatility / daily_vol

        # 口座サイズに基づくベースサイズ
        base_size = balance / price

        # ボラティリティ調整
        adjusted_size = base_size * vol_scale

        # 1000単位に丸め
        adjusted_size = max(1000, round(adjusted_size / 1000) * 1000)

        # 最大サイズ制限
        if max_size:
            adjusted_size = min(adjusted_size, max_size)

        return adjusted_size

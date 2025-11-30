"""
トレーリングストップ管理モジュール
利益を最大化するための動的ストップロス調整
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrailingMethod(Enum):
    """トレーリング方式"""
    FIXED_PIPS = "fixed_pips"      # 固定pips
    ATR_BASED = "atr_based"        # ATRベース
    PERCENT = "percent"            # パーセント
    BREAKEVEN = "breakeven"        # ブレークイーブン後トレーリング
    STEP = "step"                  # ステップ式


@dataclass
class TrailingStopConfig:
    """トレーリングストップ設定"""
    enabled: bool = True
    method: TrailingMethod = TrailingMethod.ATR_BASED

    # 固定pips設定
    trail_pips: float = 30.0

    # ATRベース設定
    atr_multiplier: float = 1.5

    # パーセント設定
    trail_percent: float = 0.005  # 0.5%

    # ブレークイーブン設定
    breakeven_trigger_pips: float = 20.0  # この利益でブレークイーブンに移動
    breakeven_offset_pips: float = 5.0    # ブレークイーブン + オフセット

    # ステップ式設定
    step_pips: float = 20.0  # このpips動くごとにSL更新

    # 共通設定
    activation_pips: float = 10.0  # この利益からトレーリング開始
    min_trail_distance: float = 15.0  # 最小トレール距離 (pips)


@dataclass
class TrailingState:
    """ポジションごとのトレーリング状態"""
    position_id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    initial_stop_loss: float
    current_stop_loss: float
    highest_price: float  # ロングの場合の最高値
    lowest_price: float   # ショートの場合の最安値
    is_activated: bool = False
    is_breakeven: bool = False
    last_update: datetime = field(default_factory=datetime.now)
    trail_history: List[Dict[str, Any]] = field(default_factory=list)


class TrailingStopManager:
    """トレーリングストップ管理クラス"""

    def __init__(self, config: Optional[TrailingStopConfig] = None):
        """
        Args:
            config: トレーリングストップ設定
        """
        self.config = config or TrailingStopConfig()
        self.states: Dict[str, TrailingState] = {}

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
    ) -> TrailingState:
        """
        ポジションを登録

        Args:
            position_id: ポジションID
            symbol: 通貨ペア
            side: 売買方向 ("long" or "short")
            entry_price: エントリー価格
            stop_loss: 初期ストップロス

        Returns:
            TrailingState
        """
        state = TrailingState(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            highest_price=entry_price,
            lowest_price=entry_price,
        )
        self.states[position_id] = state
        logger.info(f"Trailing stop registered: {position_id}, SL={stop_loss}")
        return state

    def unregister_position(self, position_id: str) -> None:
        """ポジションを登録解除"""
        if position_id in self.states:
            del self.states[position_id]
            logger.debug(f"Trailing stop unregistered: {position_id}")

    def update(
        self,
        position_id: str,
        current_price: float,
        atr: Optional[float] = None,
    ) -> Tuple[bool, Optional[float]]:
        """
        トレーリングストップを更新

        Args:
            position_id: ポジションID
            current_price: 現在価格
            atr: 現在のATR値（ATRベース方式の場合）

        Returns:
            (更新されたか, 新しいSL価格)
        """
        if not self.config.enabled:
            return False, None

        if position_id not in self.states:
            return False, None

        state = self.states[position_id]
        symbol = state.symbol
        is_long = state.side == "long"

        # pip計算用
        pip_size = 0.01 if "JPY" in symbol else 0.0001

        # 最高値/最安値を更新
        if is_long:
            state.highest_price = max(state.highest_price, current_price)
            profit_pips = (current_price - state.entry_price) / pip_size
            extreme_price = state.highest_price
        else:
            state.lowest_price = min(state.lowest_price, current_price)
            profit_pips = (state.entry_price - current_price) / pip_size
            extreme_price = state.lowest_price

        # アクティベーションチェック
        if not state.is_activated:
            if profit_pips >= self.config.activation_pips:
                state.is_activated = True
                logger.info(f"Trailing stop activated: {position_id}, profit={profit_pips:.1f} pips")
            else:
                return False, None

        # 新しいSLを計算
        new_sl = self._calculate_new_stop_loss(
            state=state,
            current_price=current_price,
            extreme_price=extreme_price,
            atr=atr,
            pip_size=pip_size,
        )

        if new_sl is None:
            return False, None

        # SLが改善されたか確認
        sl_improved = False
        if is_long:
            sl_improved = new_sl > state.current_stop_loss
        else:
            sl_improved = new_sl < state.current_stop_loss

        if sl_improved:
            old_sl = state.current_stop_loss
            state.current_stop_loss = new_sl
            state.last_update = datetime.now()

            # 履歴を記録
            state.trail_history.append({
                "timestamp": datetime.now().isoformat(),
                "price": current_price,
                "old_sl": old_sl,
                "new_sl": new_sl,
            })

            logger.info(
                f"Trailing stop updated: {position_id}, "
                f"SL {old_sl:.5f} -> {new_sl:.5f} "
                f"(price={current_price:.5f})"
            )

            return True, new_sl

        return False, None

    def _calculate_new_stop_loss(
        self,
        state: TrailingState,
        current_price: float,
        extreme_price: float,
        atr: Optional[float],
        pip_size: float,
    ) -> Optional[float]:
        """新しいストップロスを計算"""
        is_long = state.side == "long"
        method = self.config.method

        # ブレークイーブンチェック（全方式共通）
        if not state.is_breakeven:
            profit_pips = abs(current_price - state.entry_price) / pip_size
            if profit_pips >= self.config.breakeven_trigger_pips:
                state.is_breakeven = True
                offset = self.config.breakeven_offset_pips * pip_size
                if is_long:
                    return state.entry_price + offset
                else:
                    return state.entry_price - offset

        # 方式別の計算
        if method == TrailingMethod.FIXED_PIPS:
            trail_distance = self.config.trail_pips * pip_size

        elif method == TrailingMethod.ATR_BASED:
            if atr is None:
                trail_distance = self.config.trail_pips * pip_size
            else:
                trail_distance = atr * self.config.atr_multiplier

        elif method == TrailingMethod.PERCENT:
            trail_distance = extreme_price * self.config.trail_percent

        elif method == TrailingMethod.STEP:
            # ステップ式: step_pips ごとにSLを更新
            step_distance = self.config.step_pips * pip_size
            if is_long:
                steps = int((extreme_price - state.entry_price) / step_distance)
                if steps > 0:
                    return state.entry_price + (steps - 1) * step_distance
            else:
                steps = int((state.entry_price - extreme_price) / step_distance)
                if steps > 0:
                    return state.entry_price - (steps - 1) * step_distance
            return None

        else:
            trail_distance = self.config.trail_pips * pip_size

        # 最小トレール距離を適用
        min_distance = self.config.min_trail_distance * pip_size
        trail_distance = max(trail_distance, min_distance)

        # 新しいSLを計算
        if is_long:
            new_sl = extreme_price - trail_distance
        else:
            new_sl = extreme_price + trail_distance

        return new_sl

    def get_state(self, position_id: str) -> Optional[TrailingState]:
        """ポジションの状態を取得"""
        return self.states.get(position_id)

    def get_all_states(self) -> Dict[str, TrailingState]:
        """全ポジションの状態を取得"""
        return self.states.copy()

    def get_summary(self, position_id: str) -> Optional[Dict[str, Any]]:
        """ポジションのサマリーを取得"""
        state = self.states.get(position_id)
        if not state:
            return None

        is_long = state.side == "long"
        pip_size = 0.01 if "JPY" in state.symbol else 0.0001

        sl_moved_pips = abs(state.current_stop_loss - state.initial_stop_loss) / pip_size

        return {
            "position_id": position_id,
            "symbol": state.symbol,
            "side": state.side,
            "entry_price": state.entry_price,
            "initial_sl": state.initial_stop_loss,
            "current_sl": state.current_stop_loss,
            "sl_moved_pips": sl_moved_pips,
            "highest_price": state.highest_price if is_long else None,
            "lowest_price": state.lowest_price if not is_long else None,
            "is_activated": state.is_activated,
            "is_breakeven": state.is_breakeven,
            "updates_count": len(state.trail_history),
        }

    def save_states(self) -> Dict[str, Any]:
        """状態をシリアライズ（永続化用）"""
        return {
            pos_id: {
                "position_id": state.position_id,
                "symbol": state.symbol,
                "side": state.side,
                "entry_price": state.entry_price,
                "initial_stop_loss": state.initial_stop_loss,
                "current_stop_loss": state.current_stop_loss,
                "highest_price": state.highest_price,
                "lowest_price": state.lowest_price,
                "is_activated": state.is_activated,
                "is_breakeven": state.is_breakeven,
                "last_update": state.last_update.isoformat(),
            }
            for pos_id, state in self.states.items()
        }

    def load_states(self, data: Dict[str, Any]) -> None:
        """状態を復元"""
        for pos_id, state_data in data.items():
            self.states[pos_id] = TrailingState(
                position_id=state_data["position_id"],
                symbol=state_data["symbol"],
                side=state_data["side"],
                entry_price=state_data["entry_price"],
                initial_stop_loss=state_data["initial_stop_loss"],
                current_stop_loss=state_data["current_stop_loss"],
                highest_price=state_data["highest_price"],
                lowest_price=state_data["lowest_price"],
                is_activated=state_data.get("is_activated", False),
                is_breakeven=state_data.get("is_breakeven", False),
                last_update=datetime.fromisoformat(state_data["last_update"]),
            )
        logger.info(f"Loaded {len(self.states)} trailing stop states")


def load_trailing_stop_config(config_path: str = "config/risk_params.yaml") -> TrailingStopConfig:
    """設定ファイルからトレーリングストップ設定を読み込み"""
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        ts_config = config.get("trailing_stop", {})

        if not ts_config.get("enabled", True):
            return TrailingStopConfig(enabled=False)

        # メソッドの変換
        method_str = ts_config.get("method", "atr_based")
        try:
            method = TrailingMethod(method_str)
        except ValueError:
            logger.warning(f"Unknown trailing method: {method_str}, using atr_based")
            method = TrailingMethod.ATR_BASED

        return TrailingStopConfig(
            enabled=True,
            method=method,
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
        logger.warning(f"Failed to load trailing stop config: {e}, using defaults")
        return TrailingStopConfig()

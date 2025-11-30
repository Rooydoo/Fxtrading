"""
部分利確モジュール
指定した利益水準で一部ポジションを決済
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PartialCloseLevel:
    """部分利確レベル"""
    trigger_pips: float        # この利益で発動
    close_percent: float       # 決済する割合 (0.0-1.0)
    move_sl_to_entry: bool = False  # SLをエントリー価格に移動するか
    description: str = ""


@dataclass
class PartialCloseConfig:
    """部分利確設定"""
    enabled: bool = True
    levels: List[PartialCloseLevel] = field(default_factory=list)
    min_remaining_size: float = 1000  # 最小残りサイズ

    @classmethod
    def default_config(cls) -> "PartialCloseConfig":
        """デフォルト設定を返す"""
        return cls(
            enabled=True,
            levels=[
                PartialCloseLevel(
                    trigger_pips=30,
                    close_percent=0.5,
                    move_sl_to_entry=True,
                    description="30pips利益で50%決済、SLをエントリーに移動",
                ),
                PartialCloseLevel(
                    trigger_pips=50,
                    close_percent=0.5,  # 残りの50% = 全体の25%
                    move_sl_to_entry=False,
                    description="50pips利益でさらに50%決済",
                ),
            ],
            min_remaining_size=1000,
        )


@dataclass
class PartialCloseState:
    """ポジションごとの部分利確状態"""
    position_id: str
    symbol: str
    side: str
    entry_price: float
    original_size: float
    current_size: float
    closed_levels: List[int] = field(default_factory=list)  # 決済済みレベルのインデックス
    partial_close_history: List[Dict[str, Any]] = field(default_factory=list)


class PartialCloseManager:
    """部分利確管理クラス"""

    def __init__(self, config: Optional[PartialCloseConfig] = None):
        """
        Args:
            config: 部分利確設定
        """
        self.config = config or PartialCloseConfig.default_config()
        self.states: Dict[str, PartialCloseState] = {}

    def register_position(
        self,
        position_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
    ) -> PartialCloseState:
        """
        ポジションを登録

        Args:
            position_id: ポジションID
            symbol: 通貨ペア
            side: 売買方向
            entry_price: エントリー価格
            size: ポジションサイズ

        Returns:
            PartialCloseState
        """
        state = PartialCloseState(
            position_id=position_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            original_size=size,
            current_size=size,
        )
        self.states[position_id] = state
        logger.info(f"Partial close registered: {position_id}, size={size}")
        return state

    def unregister_position(self, position_id: str) -> None:
        """ポジションを登録解除"""
        if position_id in self.states:
            del self.states[position_id]

    def check_and_close(
        self,
        position_id: str,
        current_price: float,
    ) -> List[Dict[str, Any]]:
        """
        部分利確をチェックして実行すべき決済を返す

        Args:
            position_id: ポジションID
            current_price: 現在価格

        Returns:
            部分決済情報リスト [{size, reason, move_sl_to_entry}]
        """
        if not self.config.enabled:
            return []

        if position_id not in self.states:
            return []

        state = self.states[position_id]
        is_long = state.side == "long"

        # 現在の利益をpipsで計算
        pip_size = 0.01 if "JPY" in state.symbol else 0.0001

        if is_long:
            profit_pips = (current_price - state.entry_price) / pip_size
        else:
            profit_pips = (state.entry_price - current_price) / pip_size

        # 利益がない場合はスキップ
        if profit_pips <= 0:
            return []

        partial_closes = []

        # 各レベルをチェック
        for i, level in enumerate(self.config.levels):
            # すでに決済済みのレベルはスキップ
            if i in state.closed_levels:
                continue

            # トリガー条件を満たしているか
            if profit_pips >= level.trigger_pips:
                # 決済サイズを計算
                close_size = state.current_size * level.close_percent
                close_size = round(close_size / 1000) * 1000  # 1000単位に丸め

                # 最小残りサイズチェック
                remaining = state.current_size - close_size
                if remaining < self.config.min_remaining_size and remaining > 0:
                    # 全決済にするか、このレベルをスキップ
                    close_size = state.current_size

                if close_size > 0:
                    partial_closes.append({
                        "level_index": i,
                        "size": close_size,
                        "trigger_pips": level.trigger_pips,
                        "reason": f"partial_close_level_{i+1}",
                        "move_sl_to_entry": level.move_sl_to_entry,
                        "description": level.description,
                    })

        return partial_closes

    def record_partial_close(
        self,
        position_id: str,
        level_index: int,
        closed_size: float,
        close_price: float,
        pnl: float,
    ) -> None:
        """
        部分決済を記録

        Args:
            position_id: ポジションID
            level_index: 決済レベルのインデックス
            closed_size: 決済したサイズ
            close_price: 決済価格
            pnl: 実現損益
        """
        if position_id not in self.states:
            return

        state = self.states[position_id]
        state.closed_levels.append(level_index)
        state.current_size -= closed_size

        state.partial_close_history.append({
            "timestamp": datetime.now().isoformat(),
            "level_index": level_index,
            "closed_size": closed_size,
            "close_price": close_price,
            "pnl": pnl,
            "remaining_size": state.current_size,
        })

        logger.info(
            f"Partial close recorded: {position_id}, "
            f"closed={closed_size}, remaining={state.current_size}, pnl={pnl:.2f}"
        )

    def get_state(self, position_id: str) -> Optional[PartialCloseState]:
        """ポジションの状態を取得"""
        return self.states.get(position_id)

    def get_summary(self, position_id: str) -> Optional[Dict[str, Any]]:
        """ポジションのサマリーを取得"""
        state = self.states.get(position_id)
        if not state:
            return None

        total_closed = state.original_size - state.current_size
        total_pnl = sum(h.get("pnl", 0) for h in state.partial_close_history)

        return {
            "position_id": position_id,
            "original_size": state.original_size,
            "current_size": state.current_size,
            "total_closed": total_closed,
            "closed_percent": total_closed / state.original_size if state.original_size > 0 else 0,
            "partial_closes_count": len(state.partial_close_history),
            "total_realized_pnl": total_pnl,
        }

    def save_states(self) -> Dict[str, Any]:
        """状態をシリアライズ"""
        return {
            pos_id: {
                "position_id": state.position_id,
                "symbol": state.symbol,
                "side": state.side,
                "entry_price": state.entry_price,
                "original_size": state.original_size,
                "current_size": state.current_size,
                "closed_levels": state.closed_levels,
                "partial_close_history": state.partial_close_history,
            }
            for pos_id, state in self.states.items()
        }

    def load_states(self, data: Dict[str, Any]) -> None:
        """状態を復元"""
        for pos_id, state_data in data.items():
            self.states[pos_id] = PartialCloseState(
                position_id=state_data["position_id"],
                symbol=state_data["symbol"],
                side=state_data["side"],
                entry_price=state_data["entry_price"],
                original_size=state_data["original_size"],
                current_size=state_data["current_size"],
                closed_levels=state_data.get("closed_levels", []),
                partial_close_history=state_data.get("partial_close_history", []),
            )
        logger.info(f"Loaded {len(self.states)} partial close states")


def load_partial_close_config(config_path: str = "config/risk_params.yaml") -> PartialCloseConfig:
    """設定ファイルから部分利確設定を読み込み"""
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pc_config = config.get("partial_close", {})

        if not pc_config.get("enabled", True):
            return PartialCloseConfig(enabled=False)

        levels = []
        for level_data in pc_config.get("levels", []):
            levels.append(PartialCloseLevel(
                trigger_pips=level_data.get("trigger_pips", 30),
                close_percent=level_data.get("close_percent", 0.5),
                move_sl_to_entry=level_data.get("move_sl_to_entry", False),
                description=level_data.get("description", ""),
            ))

        if not levels:
            # デフォルトレベル
            levels = PartialCloseConfig.default_config().levels

        return PartialCloseConfig(
            enabled=True,
            levels=levels,
            min_remaining_size=pc_config.get("min_remaining_size", 1000),
        )

    except Exception as e:
        logger.warning(f"Failed to load partial close config: {e}, using defaults")
        return PartialCloseConfig.default_config()

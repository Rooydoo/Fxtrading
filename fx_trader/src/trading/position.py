"""
ポジション管理モジュール
建玉の追跡、損益計算、履歴管理
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class Side(Enum):
    """売買方向"""
    LONG = "BUY"
    SHORT = "SELL"


class PositionStatus(Enum):
    """ポジションステータス"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"


@dataclass
class Position:
    """ポジション情報"""
    id: str
    symbol: str
    side: Side
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN

    # 決済情報
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    # メタデータ
    confidence: float = 0.0
    max_loss_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_pnl(self, current_price: float) -> float:
        """
        損益計算

        Args:
            current_price: 現在価格

        Returns:
            損益 (通貨単位)
        """
        if self.side == Side.LONG:
            pnl = (current_price - self.entry_price) * self.size
        else:
            pnl = (self.entry_price - current_price) * self.size
        return pnl

    def calculate_pnl_pips(self, current_price: float) -> float:
        """
        損益をpipsで計算

        Args:
            current_price: 現在価格

        Returns:
            損益 (pips)
        """
        if self.side == Side.LONG:
            diff = current_price - self.entry_price
        else:
            diff = self.entry_price - current_price

        # JPYペアの場合
        if "JPY" in self.symbol:
            return diff * 100  # 0.01 = 1 pip
        else:
            return diff * 10000  # 0.0001 = 1 pip

    def calculate_pnl_percent(self, current_price: float) -> float:
        """
        損益率を計算

        Args:
            current_price: 現在価格

        Returns:
            損益率
        """
        if self.side == Side.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def should_stop_loss(self, current_price: float) -> bool:
        """ストップロス判定"""
        if self.stop_loss is None:
            return False

        if self.side == Side.LONG:
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        """テイクプロフィット判定"""
        if self.take_profit is None:
            return False

        if self.side == Side.LONG:
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit

    def close(
        self,
        exit_price: float,
        reason: str = "manual",
    ) -> float:
        """
        ポジションをクローズ

        Args:
            exit_price: 決済価格
            reason: 決済理由

        Returns:
            損益
        """
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED

        return self.calculate_pnl(exit_price)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "confidence": self.confidence,
            "max_loss_amount": self.max_loss_amount,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """辞書から生成"""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            side=Side(data["side"]),
            size=data["size"],
            entry_price=data["entry_price"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            status=PositionStatus(data.get("status", "open")),
            exit_price=data.get("exit_price"),
            exit_time=datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None,
            exit_reason=data.get("exit_reason"),
            confidence=data.get("confidence", 0.0),
            max_loss_amount=data.get("max_loss_amount", 0.0),
            metadata=data.get("metadata", {}),
        )


class PositionManager:
    """ポジション管理クラス"""

    def __init__(
        self,
        max_positions: int = 3,
        max_positions_per_pair: int = 1,
    ):
        """
        Args:
            max_positions: 最大同時ポジション数
            max_positions_per_pair: 通貨ペアあたり最大ポジション
        """
        self.max_positions = max_positions
        self.max_positions_per_pair = max_positions_per_pair
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self._position_counter = 0

    def can_open_position(self, symbol: str) -> bool:
        """
        新規ポジションを開けるか判定

        Args:
            symbol: 通貨ペア

        Returns:
            開設可能か
        """
        open_positions = self.get_open_positions()

        # 最大ポジション数チェック
        if len(open_positions) >= self.max_positions:
            return False

        # 通貨ペアあたりの最大数チェック
        pair_positions = [p for p in open_positions if p.symbol == symbol]
        if len(pair_positions) >= self.max_positions_per_pair:
            return False

        return True

    def open_position(
        self,
        symbol: str,
        side: Side,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        confidence: float = 0.0,
        max_loss_amount: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Position]:
        """
        新規ポジションを開設

        Args:
            symbol: 通貨ペア
            side: 売買方向
            size: サイズ
            entry_price: エントリー価格
            stop_loss: ストップロス
            take_profit: テイクプロフィット
            confidence: 確信度
            max_loss_amount: 最大損失額
            metadata: メタデータ

        Returns:
            作成されたPosition (失敗時はNone)
        """
        if not self.can_open_position(symbol):
            logger.warning(f"Cannot open position for {symbol}: limit reached")
            return None

        self._position_counter += 1
        position_id = f"POS_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._position_counter}"

        position = Position(
            id=position_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            max_loss_amount=max_loss_amount,
            metadata=metadata or {},
        )

        self.positions[position_id] = position
        logger.info(f"Position opened: {position_id} {symbol} {side.value} {size} @ {entry_price}")

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[float]:
        """
        ポジションを決済

        Args:
            position_id: ポジションID
            exit_price: 決済価格
            reason: 決済理由

        Returns:
            損益 (ポジションが見つからない場合はNone)
        """
        if position_id not in self.positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        position = self.positions[position_id]
        pnl = position.close(exit_price, reason)

        # クローズ済みリストに移動
        self.closed_positions.append(position)
        del self.positions[position_id]

        logger.info(f"Position closed: {position_id} PnL={pnl:.2f} Reason={reason}")

        return pnl

    def get_open_positions(self) -> List[Position]:
        """オープンポジションを取得"""
        return list(self.positions.values())

    def get_position(self, position_id: str) -> Optional[Position]:
        """特定のポジションを取得"""
        return self.positions.get(position_id)

    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """通貨ペアでポジションをフィルタ"""
        return [p for p in self.positions.values() if p.symbol == symbol]

    def check_sl_tp(self, prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        全ポジションのSL/TPをチェック

        Args:
            prices: 通貨ペアごとの現在価格

        Returns:
            決済が必要なポジション情報リスト
        """
        to_close = []

        for position in self.get_open_positions():
            current_price = prices.get(position.symbol)
            if current_price is None:
                continue

            if position.should_stop_loss(current_price):
                to_close.append({
                    "position_id": position.id,
                    "reason": "stop_loss",
                    "price": current_price,
                })
            elif position.should_take_profit(current_price):
                to_close.append({
                    "position_id": position.id,
                    "reason": "take_profit",
                    "price": current_price,
                })

        return to_close

    def get_total_exposure(self) -> float:
        """総エクスポージャーを計算"""
        return sum(p.size * p.entry_price for p in self.get_open_positions())

    def get_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        未実現損益を計算

        Args:
            prices: 現在価格辞書

        Returns:
            未実現損益合計
        """
        total = 0.0
        for position in self.get_open_positions():
            current_price = prices.get(position.symbol)
            if current_price:
                total += position.calculate_pnl(current_price)
        return total


class TradeHistory:
    """取引履歴管理 (SQLite)"""

    def __init__(self, db_path: str = "data/trades.db"):
        """
        Args:
            db_path: データベースパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                exit_reason TEXT,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                max_loss_amount REAL,
                pnl REAL,
                pnl_pips REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def save_trade(self, position: Position) -> None:
        """
        取引を保存

        Args:
            position: クローズ済みポジション
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pnl = position.calculate_pnl(position.exit_price) if position.exit_price else None
        pnl_pips = position.calculate_pnl_pips(position.exit_price) if position.exit_price else None

        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (id, symbol, side, size, entry_price, entry_time, exit_price, exit_time,
             exit_reason, stop_loss, take_profit, confidence, max_loss_amount, pnl, pnl_pips, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            position.id,
            position.symbol,
            position.side.value,
            position.size,
            position.entry_price,
            position.entry_time.isoformat(),
            position.exit_price,
            position.exit_time.isoformat() if position.exit_time else None,
            position.exit_reason,
            position.stop_loss,
            position.take_profit,
            position.confidence,
            position.max_loss_amount,
            pnl,
            pnl_pips,
            json.dumps(position.metadata),
        ))

        conn.commit()
        conn.close()

        logger.debug(f"Trade saved: {position.id}")

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        取引履歴を取得

        Args:
            symbol: フィルタする通貨ペア
            start_date: 開始日
            end_date: 終了日
            limit: 取得件数

        Returns:
            取引リスト
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        取引統計を取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            統計情報
        """
        trades = self.get_trades(start_date=start_date, end_date=end_date, limit=10000)

        if not trades:
            return {}

        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": sum(pnls),
            "average_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "average_win": sum(wins) / len(wins) if wins else 0,
            "average_loss": sum(losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "max_win": max(wins) if wins else 0,
            "max_loss": min(losses) if losses else 0,
        }

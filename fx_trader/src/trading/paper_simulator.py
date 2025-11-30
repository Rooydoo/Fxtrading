"""
ペーパートレードシミュレーター
仮想予算でのテスト運用、挙動確認、パフォーマンス分析
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PaperTradingSimulator:
    """ペーパートレードシミュレーター"""

    def __init__(
        self,
        initial_balance: float = 1000000,
        data_dir: str = "data/paper_trading",
    ):
        """
        Args:
            initial_balance: 初期仮想資金
            data_dir: データ保存ディレクトリ
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 取引履歴
        self.trades: List[Dict[str, Any]] = []
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        # 日次記録
        self.daily_records: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

        # 統計
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": initial_balance,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

        # 状態復元
        self._load_state()

    def _load_state(self) -> None:
        """状態を復元"""
        state_file = self.data_dir / "simulator_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.balance = state.get("balance", self.initial_balance)
                    self.trades = state.get("trades", [])
                    self.open_positions = state.get("open_positions", {})
                    self.stats = state.get("stats", self.stats)
                    self.daily_records = state.get("daily_records", [])
                    logger.info(f"Simulator state restored. Balance: ¥{self.balance:,.0f}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self) -> None:
        """状態を保存"""
        state = {
            "balance": self.balance,
            "trades": self.trades[-1000:],  # 直近1000件
            "open_positions": self.open_positions,
            "stats": self.stats,
            "daily_records": self.daily_records[-365:],  # 直近1年
            "saved_at": datetime.now().isoformat(),
        }
        state_file = self.data_dir / "simulator_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        ポジションを開く

        Args:
            symbol: 通貨ペア
            side: BUY or SELL
            size: サイズ
            entry_price: エントリー価格
            stop_loss: ストップロス
            take_profit: テイクプロフィット
            confidence: 確信度
            metadata: メタデータ

        Returns:
            ポジション情報
        """
        position_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        # スリッページシミュレーション
        slippage = self._simulate_slippage(symbol)
        if side == "BUY":
            actual_entry = entry_price + slippage
        else:
            actual_entry = entry_price - slippage

        position = {
            "id": position_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": actual_entry,
            "entry_time": datetime.now().isoformat(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": confidence,
            "unrealized_pnl": 0.0,
            "metadata": metadata or {},
        }

        self.open_positions[position_id] = position

        logger.info(f"[PAPER] Position opened: {position_id} {symbol} {side} {size} @ {actual_entry:.5f}")

        self.save_state()
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        """
        ポジションを閉じる

        Args:
            position_id: ポジションID
            exit_price: 決済価格
            reason: 決済理由

        Returns:
            取引結果
        """
        if position_id not in self.open_positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        position = self.open_positions[position_id]

        # スリッページシミュレーション
        slippage = self._simulate_slippage(position["symbol"])
        if position["side"] == "BUY":
            actual_exit = exit_price - slippage
        else:
            actual_exit = exit_price + slippage

        # PnL計算
        if position["side"] == "BUY":
            pnl = (actual_exit - position["entry_price"]) * position["size"]
        else:
            pnl = (position["entry_price"] - actual_exit) * position["size"]

        # 手数料
        commission = abs(pnl) * 0.00002  # 0.002%
        pnl -= commission

        # 残高更新
        self.balance += pnl

        # 取引記録
        trade = {
            "position_id": position_id,
            "symbol": position["symbol"],
            "side": position["side"],
            "size": position["size"],
            "entry_price": position["entry_price"],
            "entry_time": position["entry_time"],
            "exit_price": actual_exit,
            "exit_time": datetime.now().isoformat(),
            "exit_reason": reason,
            "pnl": pnl,
            "commission": commission,
            "confidence": position["confidence"],
            "balance_after": self.balance,
        }
        self.trades.append(trade)

        # 統計更新
        self._update_stats(pnl)

        # 株式曲線記録
        self.equity_curve.append((datetime.now(), self.balance))

        # ポジション削除
        del self.open_positions[position_id]

        logger.info(f"[PAPER] Position closed: {position_id} PnL=¥{pnl:+,.0f} Reason={reason}")

        self.save_state()
        return trade

    def check_sl_tp(self, prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        全ポジションのSL/TPをチェック

        Args:
            prices: 現在価格辞書 {symbol: price}

        Returns:
            決済されたポジションリスト
        """
        closed = []

        for pos_id, pos in list(self.open_positions.items()):
            price = prices.get(pos["symbol"])
            if price is None:
                continue

            # 未実現損益更新
            if pos["side"] == "BUY":
                pos["unrealized_pnl"] = (price - pos["entry_price"]) * pos["size"]
            else:
                pos["unrealized_pnl"] = (pos["entry_price"] - price) * pos["size"]

            # SL/TPチェック
            if pos["side"] == "BUY":
                if price <= pos["stop_loss"]:
                    result = self.close_position(pos_id, pos["stop_loss"], "stop_loss")
                    if result:
                        closed.append(result)
                elif price >= pos["take_profit"]:
                    result = self.close_position(pos_id, pos["take_profit"], "take_profit")
                    if result:
                        closed.append(result)
            else:  # SELL
                if price >= pos["stop_loss"]:
                    result = self.close_position(pos_id, pos["stop_loss"], "stop_loss")
                    if result:
                        closed.append(result)
                elif price <= pos["take_profit"]:
                    result = self.close_position(pos_id, pos["take_profit"], "take_profit")
                    if result:
                        closed.append(result)

        return closed

    def _simulate_slippage(self, symbol: str) -> float:
        """スリッページをシミュレート"""
        # 正規分布でスリッページを生成 (平均0.5pips)
        if "JPY" in symbol:
            return np.random.normal(0.005, 0.002)  # 0.5 pips ± 0.2
        else:
            return np.random.normal(0.00005, 0.00002)

    def _update_stats(self, pnl: float) -> None:
        """統計を更新"""
        self.stats["total_trades"] += 1
        self.stats["total_pnl"] += pnl

        if pnl > 0:
            self.stats["winning_trades"] += 1
            self.stats["consecutive_wins"] += 1
            self.stats["consecutive_losses"] = 0
            if self.stats["consecutive_wins"] > self.stats["max_consecutive_wins"]:
                self.stats["max_consecutive_wins"] = self.stats["consecutive_wins"]
        else:
            self.stats["losing_trades"] += 1
            self.stats["consecutive_losses"] += 1
            self.stats["consecutive_wins"] = 0
            if self.stats["consecutive_losses"] > self.stats["max_consecutive_losses"]:
                self.stats["max_consecutive_losses"] = self.stats["consecutive_losses"]

        # ドローダウン
        if self.balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = self.balance
        drawdown = (self.stats["peak_balance"] - self.balance) / self.stats["peak_balance"]
        if drawdown > self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = drawdown

    def record_daily(self) -> None:
        """日次記録"""
        today = datetime.now().date()

        # 今日の取引を集計
        today_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t["exit_time"]).date() == today
        ]

        daily = {
            "date": today.isoformat(),
            "trades": len(today_trades),
            "pnl": sum(t["pnl"] for t in today_trades),
            "balance": self.balance,
            "open_positions": len(self.open_positions),
            "win_rate": (
                len([t for t in today_trades if t["pnl"] > 0]) / len(today_trades)
                if today_trades else 0
            ),
        }

        self.daily_records.append(daily)
        self.save_state()

    def get_status(self) -> Dict[str, Any]:
        """現在のステータスを取得"""
        # 未実現損益合計
        unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in self.open_positions.values())

        # 勝率
        win_rate = (
            self.stats["winning_trades"] / self.stats["total_trades"]
            if self.stats["total_trades"] > 0 else 0
        )

        # リターン
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        # シャープレシオ (簡易計算)
        if len(self.trades) > 1:
            pnls = [t["pnl"] for t in self.trades[-100:]]  # 直近100取引
            if np.std(pnls) > 0:
                sharpe = np.sqrt(252) * np.mean(pnls) / np.std(pnls)
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            "mode": "PAPER TRADING",
            "initial_balance": self.initial_balance,
            "current_balance": self.balance,
            "unrealized_pnl": unrealized_pnl,
            "equity": self.balance + unrealized_pnl,
            "total_return": total_return,
            "total_pnl": self.stats["total_pnl"],
            "total_trades": self.stats["total_trades"],
            "win_rate": win_rate,
            "max_drawdown": self.stats["max_drawdown"],
            "sharpe_ratio": sharpe,
            "open_positions": len(self.open_positions),
            "consecutive_wins": self.stats["consecutive_wins"],
            "consecutive_losses": self.stats["consecutive_losses"],
            "positions": list(self.open_positions.values()),
        }

    def get_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        パフォーマンスレポートを生成

        Args:
            days: 対象日数

        Returns:
            レポート
        """
        cutoff = datetime.now() - timedelta(days=days)

        recent_trades = [
            t for t in self.trades
            if datetime.fromisoformat(t["exit_time"]) >= cutoff
        ]

        if not recent_trades:
            return {"period": days, "no_data": True}

        pnls = [t["pnl"] for t in recent_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # 時間帯別分析
        hourly_pnl = {}
        for t in recent_trades:
            hour = datetime.fromisoformat(t["entry_time"]).hour
            if hour not in hourly_pnl:
                hourly_pnl[hour] = []
            hourly_pnl[hour].append(t["pnl"])

        best_hour = max(hourly_pnl.keys(), key=lambda h: sum(hourly_pnl[h])) if hourly_pnl else None
        worst_hour = min(hourly_pnl.keys(), key=lambda h: sum(hourly_pnl[h])) if hourly_pnl else None

        # 通貨ペア別
        symbol_pnl = {}
        for t in recent_trades:
            sym = t["symbol"]
            if sym not in symbol_pnl:
                symbol_pnl[sym] = []
            symbol_pnl[sym].append(t["pnl"])

        return {
            "period_days": days,
            "total_trades": len(recent_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(recent_trades) if recent_trades else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf"),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "best_hour": best_hour,
            "worst_hour": worst_hour,
            "symbol_performance": {
                sym: {"trades": len(p), "pnl": sum(p), "win_rate": len([x for x in p if x > 0]) / len(p)}
                for sym, p in symbol_pnl.items()
            },
        }

    def reset(self, confirm: bool = False) -> bool:
        """
        シミュレーターをリセット

        Args:
            confirm: 確認フラグ

        Returns:
            成功フラグ
        """
        if not confirm:
            logger.warning("Reset requires confirm=True")
            return False

        # バックアップ
        backup_file = self.data_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        state = {
            "balance": self.balance,
            "trades": self.trades,
            "stats": self.stats,
        }
        with open(backup_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        # リセット
        self.balance = self.initial_balance
        self.trades = []
        self.open_positions = {}
        self.daily_records = []
        self.equity_curve = []
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": self.initial_balance,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

        self.save_state()
        logger.info(f"Simulator reset. Backup saved to {backup_file}")
        return True

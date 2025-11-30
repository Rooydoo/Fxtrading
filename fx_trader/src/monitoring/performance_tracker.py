"""
パフォーマンストラッキングモジュール
リアルタイムのパフォーマンス監視と統計計算
"""
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """パフォーマンストラッカー"""

    def __init__(
        self,
        initial_balance: float = 1000000,
        window_size: int = 100,
    ):
        """
        Args:
            initial_balance: 初期残高
            window_size: 移動ウィンドウサイズ
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.window_size = window_size

        # 取引履歴
        self.trades: List[Dict[str, Any]] = []
        self.recent_pnls = deque(maxlen=window_size)
        self.daily_pnls: Dict[str, float] = {}

        # 統計
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

    def record_trade(
        self,
        pnl: float,
        symbol: str,
        side: str,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """
        取引を記録

        Args:
            pnl: 損益
            symbol: 通貨ペア
            side: 売買方向
            entry_time: エントリー時刻
            exit_time: 決済時刻
        """
        trade = {
            "pnl": pnl,
            "symbol": symbol,
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "recorded_at": datetime.now(),
        }

        self.trades.append(trade)
        self.recent_pnls.append(pnl)

        # 残高更新
        self.current_balance += pnl

        # ピーク更新とドローダウン計算
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        current_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        # 連勝・連敗カウント
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if self.consecutive_wins > self.max_consecutive_wins:
                self.max_consecutive_wins = self.consecutive_wins
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if self.consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.consecutive_losses

        # 日次損益記録
        date_key = exit_time.strftime("%Y-%m-%d")
        if date_key not in self.daily_pnls:
            self.daily_pnls[date_key] = 0
        self.daily_pnls[date_key] += pnl

        logger.debug(f"Trade recorded: PnL={pnl:.2f}, Balance={self.current_balance:.2f}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        現在のパフォーマンス指標を取得

        Returns:
            パフォーマンス指標辞書
        """
        if not self.trades:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # 基本統計
        total_pnl = sum(pnls)
        total_return = total_pnl / self.initial_balance

        win_rate = len(wins) / len(pnls) if pnls else 0

        # 平均損益
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        # プロフィットファクター
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # シャープレシオ
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_balance
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        else:
            sharpe = 0

        # 期待値
        expectancy = np.mean(pnls) if pnls else 0

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "max_drawdown": self.max_drawdown,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "expectancy": expectancy,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """空の指標を返す"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_return": 0,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "max_drawdown": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "expectancy": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    def get_recent_metrics(self, n: int = 20) -> Dict[str, Any]:
        """
        直近N取引の指標を取得

        Args:
            n: 取引数

        Returns:
            パフォーマンス指標
        """
        if len(self.trades) < n:
            n = len(self.trades)

        if n == 0:
            return self._empty_metrics()

        recent_trades = self.trades[-n:]
        pnls = [t["pnl"] for t in recent_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "n_trades": n,
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "sharpe_ratio": np.sqrt(252) * np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0,
        }

    def get_period_metrics(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        期間別の指標を取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            パフォーマンス指標
        """
        if end_date is None:
            end_date = datetime.now()

        period_trades = [
            t for t in self.trades
            if start_date <= t["exit_time"] <= end_date
        ]

        if not period_trades:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in period_trades]
        wins = [p for p in pnls if p > 0]

        return {
            "period": f"{start_date.date()} - {end_date.date()}",
            "total_trades": len(period_trades),
            "win_rate": len(wins) / len(pnls) if pnls else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
        }

    def get_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        通貨ペア別パフォーマンスを取得

        Returns:
            通貨ペアごとの指標
        """
        symbol_trades: Dict[str, List] = {}

        for trade in self.trades:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade["pnl"])

        results = {}
        for symbol, pnls in symbol_trades.items():
            wins = [p for p in pnls if p > 0]
            results[symbol] = {
                "total_trades": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": np.mean(pnls),
            }

        return results

    def get_hourly_performance(self) -> Dict[int, Dict[str, Any]]:
        """
        時間帯別パフォーマンスを取得

        Returns:
            時間帯ごとの指標
        """
        hourly_trades: Dict[int, List] = {h: [] for h in range(24)}

        for trade in self.trades:
            hour = trade["entry_time"].hour
            hourly_trades[hour].append(trade["pnl"])

        results = {}
        for hour, pnls in hourly_trades.items():
            if pnls:
                wins = [p for p in pnls if p > 0]
                results[hour] = {
                    "total_trades": len(pnls),
                    "win_rate": len(wins) / len(pnls),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                }
            else:
                results[hour] = {
                    "total_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                }

        return results

    def reset(self) -> None:
        """トラッカーをリセット"""
        self.trades.clear()
        self.recent_pnls.clear()
        self.daily_pnls.clear()
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0

        logger.info("Performance tracker reset")

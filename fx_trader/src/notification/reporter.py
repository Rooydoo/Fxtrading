"""
レポート生成モジュール
日次・週次・月次レポートの生成
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ..trading.position import TradeHistory
from .telegram import TelegramNotifier

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """パフォーマンスレポート生成"""

    def __init__(
        self,
        trade_history: TradeHistory,
        notifier: TelegramNotifier,
    ):
        """
        Args:
            trade_history: 取引履歴
            notifier: Telegram通知
        """
        self.trade_history = trade_history
        self.notifier = notifier

    def generate_daily_report(
        self,
        date: Optional[datetime] = None,
        open_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        日次レポートを生成

        Args:
            date: 対象日 (省略時は今日)
            open_positions: オープンポジション

        Returns:
            レポートデータ
        """
        if date is None:
            date = datetime.now()

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        trades = self.trade_history.get_trades(
            start_date=start,
            end_date=end,
            limit=1000,
        )

        # 基本統計
        trades_count = len(trades)
        wins = len([t for t in trades if t.get("pnl", 0) > 0])
        losses = len([t for t in trades if t.get("pnl", 0) <= 0])
        total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)

        # 累計損益 (全期間)
        all_stats = self.trade_history.get_statistics()
        cumulative_pnl = all_stats.get("total_pnl", 0)

        report = {
            "date": start.strftime("%Y-%m-%d"),
            "trades_count": trades_count,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades_count if trades_count > 0 else 0,
            "total_pnl": total_pnl,
            "cumulative_pnl": cumulative_pnl,
            "positions": open_positions or [],
        }

        return report

    def generate_weekly_report(
        self,
        week_start: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        週次レポートを生成

        Args:
            week_start: 週開始日 (省略時は先週月曜日)

        Returns:
            レポートデータ
        """
        if week_start is None:
            today = datetime.now()
            # 先週の月曜日
            week_start = today - timedelta(days=today.weekday() + 7)

        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)

        trades = self.trade_history.get_trades(
            start_date=week_start,
            end_date=week_end,
            limit=1000,
        )

        # 基本統計
        trades_count = len(trades)
        wins = len([t for t in trades if t.get("pnl", 0) > 0])
        win_rate = wins / trades_count if trades_count > 0 else 0

        pnls = [t.get("pnl", 0) or 0 for t in trades]
        total_pnl = sum(pnls)

        # ドローダウン計算
        if pnls:
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0

        # シャープレシオ (日次リターンの年率換算)
        if len(pnls) > 1:
            returns = np.array(pnls)
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        else:
            sharpe = 0

        # 最大利益・損失
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0

        # 時間帯別パフォーマンス
        hourly_performance = {}
        for trade in trades:
            try:
                entry_time = datetime.fromisoformat(trade["entry_time"])
                hour = entry_time.hour
                pnl = trade.get("pnl", 0) or 0
                if hour not in hourly_performance:
                    hourly_performance[hour] = 0
                hourly_performance[hour] += pnl
            except (KeyError, ValueError):
                pass

        report = {
            "week": f"{week_start.strftime('%Y-%m-%d')} - {week_end.strftime('%Y-%m-%d')}",
            "trades_count": trades_count,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "hourly_performance": hourly_performance,
        }

        return report

    def generate_monthly_report(
        self,
        month: Optional[datetime] = None,
        initial_balance: float = 1000000,
    ) -> Dict[str, Any]:
        """
        月次レポートを生成

        Args:
            month: 対象月 (省略時は先月)
            initial_balance: 月初残高

        Returns:
            レポートデータ
        """
        if month is None:
            today = datetime.now()
            month = today.replace(day=1) - timedelta(days=1)

        month_start = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)

        trades = self.trade_history.get_trades(
            start_date=month_start,
            end_date=month_end,
            limit=10000,
        )

        # 基本統計
        trades_count = len(trades)
        wins = len([t for t in trades if t.get("pnl", 0) > 0])
        win_rate = wins / trades_count if trades_count > 0 else 0

        pnls = [t.get("pnl", 0) or 0 for t in trades]
        total_pnl = sum(pnls)
        final_balance = initial_balance + total_pnl
        total_return = total_pnl / initial_balance if initial_balance > 0 else 0

        # プロフィットファクター
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # ドローダウン
        if pnls:
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max() / initial_balance if initial_balance > 0 else 0
        else:
            max_drawdown = 0

        # シャープレシオ
        if len(pnls) > 1:
            returns = np.array(pnls) / initial_balance
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        else:
            sharpe = 0

        report = {
            "month": month_start.strftime("%Y-%m"),
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return": total_return,
            "trades_count": trades_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
        }

        return report

    def send_daily_report(
        self,
        date: Optional[datetime] = None,
        open_positions: Optional[List[Dict]] = None,
    ) -> bool:
        """日次レポートを送信"""
        report = self.generate_daily_report(date, open_positions)

        return self.notifier.send_daily_report(
            date=report["date"],
            trades_count=report["trades_count"],
            wins=report["wins"],
            losses=report["losses"],
            total_pnl=report["total_pnl"],
            cumulative_pnl=report["cumulative_pnl"],
            positions=report["positions"],
        )

    def send_weekly_report(
        self,
        week_start: Optional[datetime] = None,
    ) -> bool:
        """週次レポートを送信"""
        report = self.generate_weekly_report(week_start)

        return self.notifier.send_weekly_report(
            week=report["week"],
            trades_count=report["trades_count"],
            win_rate=report["win_rate"],
            total_pnl=report["total_pnl"],
            max_drawdown=report["max_drawdown"],
            sharpe_ratio=report["sharpe_ratio"],
            best_trade=report["best_trade"],
            worst_trade=report["worst_trade"],
            hourly_performance=report.get("hourly_performance"),
        )

    def send_monthly_report(
        self,
        month: Optional[datetime] = None,
        initial_balance: float = 1000000,
    ) -> bool:
        """月次レポートを送信"""
        report = self.generate_monthly_report(month, initial_balance)

        return self.notifier.send_monthly_report(
            month=report["month"],
            initial_balance=report["initial_balance"],
            final_balance=report["final_balance"],
            total_return=report["total_return"],
            trades_count=report["trades_count"],
            win_rate=report["win_rate"],
            profit_factor=report["profit_factor"],
            max_drawdown=report["max_drawdown"],
            sharpe_ratio=report["sharpe_ratio"],
        )


class ReportScheduler:
    """レポートスケジューラー"""

    def __init__(self, reporter: PerformanceReporter):
        """
        Args:
            reporter: レポーター
        """
        self.reporter = reporter
        self._last_daily_report: Optional[datetime] = None
        self._last_weekly_report: Optional[datetime] = None
        self._last_monthly_report: Optional[datetime] = None

    def check_and_send_reports(
        self,
        open_positions: Optional[List[Dict]] = None,
        current_balance: float = 1000000,
    ) -> None:
        """
        スケジュールに従ってレポートを送信

        Args:
            open_positions: オープンポジション
            current_balance: 現在残高
        """
        now = datetime.now()

        # 日次レポート (23:55頃)
        if now.hour == 23 and now.minute >= 55:
            if self._last_daily_report is None or self._last_daily_report.date() != now.date():
                self.reporter.send_daily_report(now, open_positions)
                self._last_daily_report = now
                logger.info("Daily report sent")

        # 週次レポート (日曜日 23:00頃)
        if now.weekday() == 6 and now.hour == 23:  # Sunday
            week_start = now - timedelta(days=6)
            if self._last_weekly_report is None or self._last_weekly_report.isocalendar()[1] != now.isocalendar()[1]:
                self.reporter.send_weekly_report(week_start)
                self._last_weekly_report = now
                logger.info("Weekly report sent")

        # 月次レポート (月初1日 00:05頃)
        if now.day == 1 and now.hour == 0 and now.minute >= 5:
            last_month = now - timedelta(days=1)
            if self._last_monthly_report is None or self._last_monthly_report.month != last_month.month:
                self.reporter.send_monthly_report(last_month, current_balance)
                self._last_monthly_report = now
                logger.info("Monthly report sent")

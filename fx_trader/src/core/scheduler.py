"""
スケジューラーモジュール
15分足ベースの定期実行制御
"""
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TradingScheduler:
    """トレーディングスケジューラー"""

    def __init__(
        self,
        interval_minutes: int = 15,
        weekdays_only: bool = True,
    ):
        """
        Args:
            interval_minutes: 実行間隔 (分)
            weekdays_only: 平日のみ実行
        """
        self.interval_minutes = interval_minutes
        self.weekdays_only = weekdays_only

        self._running = False
        self._callbacks: List[Callable[[], None]] = []
        self._error_handlers: List[Callable[[Exception], None]] = []
        self._last_run: Optional[datetime] = None
        self._run_count = 0

    def add_callback(self, callback: Callable[[], None]) -> None:
        """
        実行コールバックを追加

        Args:
            callback: 定期実行する関数
        """
        self._callbacks.append(callback)

    def add_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """
        エラーハンドラーを追加

        Args:
            handler: エラー処理関数
        """
        self._error_handlers.append(handler)

    def _should_run(self) -> bool:
        """実行すべきか判定"""
        now = datetime.now()

        # 平日チェック
        if self.weekdays_only and now.weekday() >= 5:  # 土日
            return False

        # 間隔チェック
        if self._last_run is not None:
            elapsed = (now - self._last_run).total_seconds()
            if elapsed < self.interval_minutes * 60:
                return False

        return True

    def _wait_for_next_interval(self) -> None:
        """次の実行時刻まで待機"""
        now = datetime.now()

        # 次のN分の00秒を計算
        next_minute = ((now.minute // self.interval_minutes) + 1) * self.interval_minutes
        if next_minute >= 60:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_run = now.replace(minute=next_minute, second=0, microsecond=0)

        # 待機時間
        wait_seconds = (next_run - now).total_seconds()
        if wait_seconds > 0:
            logger.debug(f"Waiting {wait_seconds:.0f}s until next run at {next_run}")
            time.sleep(wait_seconds)

    def _run_callbacks(self) -> None:
        """全コールバックを実行"""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception(f"Callback error: {e}")
                for handler in self._error_handlers:
                    try:
                        handler(e)
                    except Exception as he:
                        logger.error(f"Error handler failed: {he}")

    def start(self) -> None:
        """スケジューラーを開始"""
        self._running = True
        logger.info(f"Scheduler started (interval={self.interval_minutes}min)")

        # シグナルハンドラー設定
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, stopping scheduler")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self._running:
            try:
                if self._should_run():
                    logger.info(f"Running scheduled tasks (run #{self._run_count + 1})")
                    self._run_callbacks()
                    self._last_run = datetime.now()
                    self._run_count += 1

                self._wait_for_next_interval()

            except Exception as e:
                logger.exception(f"Scheduler error: {e}")
                time.sleep(60)  # エラー時は1分待機

        logger.info("Scheduler stopped")

    def stop(self) -> None:
        """スケジューラーを停止"""
        self._running = False

    def run_once(self) -> None:
        """1回だけ実行"""
        logger.info("Running single execution")
        self._run_callbacks()
        self._last_run = datetime.now()
        self._run_count += 1

    @property
    def is_running(self) -> bool:
        """実行中かどうか"""
        return self._running

    @property
    def run_count(self) -> int:
        """実行回数"""
        return self._run_count


class MarketHoursChecker:
    """市場時間チェッカー"""

    # FX市場は平日24時間 (日曜夜～金曜夜)
    # 土曜日と日曜日の大部分は休場

    def __init__(self, timezone: str = "Asia/Tokyo"):
        """
        Args:
            timezone: タイムゾーン
        """
        self.timezone = timezone

    def is_market_open(self) -> bool:
        """
        市場がオープンしているか確認

        Returns:
            オープンフラグ
        """
        now = datetime.now()

        # 土曜日は休場
        if now.weekday() == 5:
            return False

        # 日曜日は一部時間のみ
        if now.weekday() == 6:
            # 日曜夜21:00以降はオープン (日本時間)
            return now.hour >= 21

        # 金曜日は一定時刻まで
        if now.weekday() == 4:
            # 金曜日6:00までオープン (日本時間、翌週への切り替わり)
            # 実際には金曜深夜まで開いている
            return True

        # 月-木は24時間
        return True

    def get_next_market_open(self) -> datetime:
        """
        次の市場オープン時刻を取得

        Returns:
            次のオープン時刻
        """
        now = datetime.now()

        if self.is_market_open():
            return now

        # 土曜日の場合
        if now.weekday() == 5:
            # 翌日 (日曜) の21時
            next_open = now.replace(hour=21, minute=0, second=0, microsecond=0)
            next_open += timedelta(days=1)
            return next_open

        # 日曜日の早い時間
        if now.weekday() == 6 and now.hour < 21:
            return now.replace(hour=21, minute=0, second=0, microsecond=0)

        return now

    def time_until_open(self) -> timedelta:
        """
        市場オープンまでの時間を取得

        Returns:
            残り時間
        """
        if self.is_market_open():
            return timedelta(0)

        next_open = self.get_next_market_open()
        return next_open - datetime.now()


class HealthChecker:
    """ヘルスチェッカー"""

    def __init__(self):
        self._last_heartbeat: Optional[datetime] = None
        self._errors: List[Dict[str, Any]] = []
        self._status = "unknown"

    def heartbeat(self) -> None:
        """ハートビートを記録"""
        self._last_heartbeat = datetime.now()
        self._status = "healthy"

    def record_error(self, error: Exception, context: str = "") -> None:
        """
        エラーを記録

        Args:
            error: 例外
            context: コンテキスト
        """
        self._errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
        })

        # 最大100件保持
        if len(self._errors) > 100:
            self._errors = self._errors[-100:]

        self._status = "unhealthy"

    def get_status(self) -> Dict[str, Any]:
        """
        ステータスを取得

        Returns:
            ステータス情報
        """
        now = datetime.now()

        # ハートビートが古すぎる場合
        if self._last_heartbeat:
            heartbeat_age = (now - self._last_heartbeat).total_seconds()
            if heartbeat_age > 300:  # 5分以上
                self._status = "stale"
        else:
            self._status = "unknown"

        return {
            "status": self._status,
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
            "recent_errors": self._errors[-10:],
            "total_errors": len(self._errors),
        }

    def is_healthy(self) -> bool:
        """健全か確認"""
        status = self.get_status()
        return status["status"] == "healthy"

"""
経済指標カレンダー連携モジュール
重要指標発表時の取引回避
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """イベント重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"


@dataclass
class EconomicEvent:
    """経済イベント"""
    event_id: str
    title: str
    country: str
    currency: str
    timestamp: datetime
    impact: EventImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    def affects_pair(self, symbol: str) -> bool:
        """この通貨ペアに影響するか"""
        return self.currency in symbol.replace("_", "")


class EconomicCalendar:
    """経済指標カレンダー"""

    # 無料API: investing.com, forexfactory, etc.
    # ここでは自前のJSONファイルベースの実装を提供

    def __init__(
        self,
        calendar_file: str = "data/economic_calendar.json",
        cache_hours: int = 6,
    ):
        """
        Args:
            calendar_file: カレンダーファイルパス
            cache_hours: キャッシュ有効期間（時間）
        """
        self.calendar_file = Path(calendar_file)
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours

        self._events: List[EconomicEvent] = []
        self._last_update: Optional[datetime] = None

    def load_events(self) -> List[EconomicEvent]:
        """イベントを読み込み"""
        if self.calendar_file.exists():
            try:
                with open(self.calendar_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self._events = []
                for event_data in data.get("events", []):
                    event = EconomicEvent(
                        event_id=event_data.get("id", ""),
                        title=event_data.get("title", ""),
                        country=event_data.get("country", ""),
                        currency=event_data.get("currency", ""),
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        impact=EventImpact(event_data.get("impact", "low")),
                        forecast=event_data.get("forecast"),
                        previous=event_data.get("previous"),
                        actual=event_data.get("actual"),
                    )
                    self._events.append(event)

                self._last_update = datetime.now()
                logger.info(f"Loaded {len(self._events)} economic events")

            except Exception as e:
                logger.error(f"Failed to load calendar: {e}")

        return self._events

    def save_events(self, events: List[EconomicEvent]) -> None:
        """イベントを保存"""
        data = {
            "updated_at": datetime.now().isoformat(),
            "events": [
                {
                    "id": e.event_id,
                    "title": e.title,
                    "country": e.country,
                    "currency": e.currency,
                    "timestamp": e.timestamp.isoformat(),
                    "impact": e.impact.value,
                    "forecast": e.forecast,
                    "previous": e.previous,
                    "actual": e.actual,
                }
                for e in events
            ]
        }

        with open(self.calendar_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self._events = events
        self._last_update = datetime.now()

    def get_upcoming_events(
        self,
        hours: int = 24,
        min_impact: EventImpact = EventImpact.MEDIUM,
        currencies: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """
        今後のイベントを取得

        Args:
            hours: 何時間先まで
            min_impact: 最小重要度
            currencies: 対象通貨リスト

        Returns:
            イベントリスト
        """
        if not self._events:
            self.load_events()

        now = datetime.now()
        end_time = now + timedelta(hours=hours)

        impact_order = {
            EventImpact.LOW: 1,
            EventImpact.MEDIUM: 2,
            EventImpact.HIGH: 3,
            EventImpact.HOLIDAY: 3,
        }
        min_order = impact_order.get(min_impact, 2)

        results = []
        for event in self._events:
            # 時間範囲チェック
            if not (now <= event.timestamp <= end_time):
                continue

            # 重要度チェック
            if impact_order.get(event.impact, 0) < min_order:
                continue

            # 通貨チェック
            if currencies and event.currency not in currencies:
                continue

            results.append(event)

        return sorted(results, key=lambda e: e.timestamp)

    def get_events_near_time(
        self,
        target_time: datetime,
        window_minutes: int = 30,
        min_impact: EventImpact = EventImpact.HIGH,
    ) -> List[EconomicEvent]:
        """
        特定時刻の前後のイベントを取得

        Args:
            target_time: 対象時刻
            window_minutes: 前後の分数
            min_impact: 最小重要度

        Returns:
            イベントリスト
        """
        if not self._events:
            self.load_events()

        window = timedelta(minutes=window_minutes)
        start_time = target_time - window
        end_time = target_time + window

        impact_order = {
            EventImpact.LOW: 1,
            EventImpact.MEDIUM: 2,
            EventImpact.HIGH: 3,
            EventImpact.HOLIDAY: 3,
        }
        min_order = impact_order.get(min_impact, 3)

        results = []
        for event in self._events:
            if start_time <= event.timestamp <= end_time:
                if impact_order.get(event.impact, 0) >= min_order:
                    results.append(event)

        return results


class TradingFilter:
    """取引フィルター（経済指標ベース）"""

    # 重要イベント前後の回避時間（分）
    DEFAULT_BLACKOUT_MINUTES = {
        EventImpact.HIGH: 30,
        EventImpact.MEDIUM: 15,
        EventImpact.LOW: 0,
        EventImpact.HOLIDAY: 0,
    }

    # 特に重要なイベント（より長い回避時間）
    HIGH_IMPACT_EVENTS = {
        "Non-Farm Payrolls": 60,
        "NFP": 60,
        "雇用統計": 60,
        "FOMC": 60,
        "Interest Rate Decision": 45,
        "金利決定": 45,
        "GDP": 30,
        "CPI": 30,
        "消費者物価指数": 30,
        "ECB Press Conference": 45,
    }

    def __init__(
        self,
        calendar: EconomicCalendar,
        blackout_minutes: Optional[Dict[EventImpact, int]] = None,
    ):
        """
        Args:
            calendar: EconomicCalendarインスタンス
            blackout_minutes: 重要度別の回避時間
        """
        self.calendar = calendar
        self.blackout_minutes = blackout_minutes or self.DEFAULT_BLACKOUT_MINUTES

    def can_trade(
        self,
        symbol: str,
        trade_time: Optional[datetime] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        取引可能か判定

        Args:
            symbol: 通貨ペア
            trade_time: 取引時刻（デフォルトは現在）

        Returns:
            (取引可能フラグ, 理由)
        """
        if trade_time is None:
            trade_time = datetime.now()

        # 通貨ペアから関連通貨を抽出
        currencies = self._extract_currencies(symbol)

        # 重要イベントをチェック
        for impact in [EventImpact.HIGH, EventImpact.MEDIUM]:
            blackout = self.blackout_minutes.get(impact, 0)
            if blackout == 0:
                continue

            events = self.calendar.get_events_near_time(
                target_time=trade_time,
                window_minutes=blackout,
                min_impact=impact,
            )

            for event in events:
                # この通貨ペアに影響するイベントか
                if event.currency in currencies:
                    # 特別重要イベントの追加チェック
                    extra_blackout = self._get_extra_blackout(event.title)
                    if extra_blackout > 0:
                        events_extra = self.calendar.get_events_near_time(
                            target_time=trade_time,
                            window_minutes=extra_blackout,
                            min_impact=impact,
                        )
                        if event in events_extra:
                            return False, f"High impact event: {event.title} ({event.currency})"

                    return False, f"Economic event: {event.title} ({event.currency})"

        return True, None

    def _extract_currencies(self, symbol: str) -> Set[str]:
        """通貨ペアから通貨を抽出"""
        symbol = symbol.replace("_", "").upper()
        currencies = set()

        # 標準的な通貨コード
        known_currencies = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]

        for curr in known_currencies:
            if curr in symbol:
                currencies.add(curr)

        return currencies

    def _get_extra_blackout(self, title: str) -> int:
        """特別重要イベントの追加回避時間"""
        title_upper = title.upper()
        for event_name, minutes in self.HIGH_IMPACT_EVENTS.items():
            if event_name.upper() in title_upper:
                return minutes
        return 0

    def get_blocked_periods(
        self,
        symbol: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        今後のブロック期間を取得

        Args:
            symbol: 通貨ペア
            hours: 何時間先まで

        Returns:
            ブロック期間リスト
        """
        currencies = self._extract_currencies(symbol)

        events = self.calendar.get_upcoming_events(
            hours=hours,
            min_impact=EventImpact.MEDIUM,
            currencies=list(currencies),
        )

        blocked_periods = []
        for event in events:
            blackout = self.blackout_minutes.get(event.impact, 0)
            extra = self._get_extra_blackout(event.title)
            total_blackout = max(blackout, extra)

            if total_blackout > 0:
                blocked_periods.append({
                    "event": event.title,
                    "currency": event.currency,
                    "impact": event.impact.value,
                    "event_time": event.timestamp.isoformat(),
                    "blocked_from": (event.timestamp - timedelta(minutes=total_blackout)).isoformat(),
                    "blocked_until": (event.timestamp + timedelta(minutes=total_blackout)).isoformat(),
                })

        return blocked_periods


class CalendarUpdater:
    """カレンダー更新クラス（外部API連携）"""

    # Investing.com の無料API（非公式）
    INVESTING_API = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"

    def __init__(self, calendar: EconomicCalendar):
        """
        Args:
            calendar: EconomicCalendarインスタンス
        """
        self.calendar = calendar

    def update_from_manual(self, events_data: List[Dict[str, Any]]) -> int:
        """
        手動でイベントを追加

        Args:
            events_data: イベントデータリスト

        Returns:
            追加したイベント数
        """
        events = []
        for i, data in enumerate(events_data):
            try:
                event = EconomicEvent(
                    event_id=data.get("id", f"manual_{i}"),
                    title=data["title"],
                    country=data.get("country", ""),
                    currency=data["currency"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    impact=EventImpact(data.get("impact", "medium")),
                    forecast=data.get("forecast"),
                    previous=data.get("previous"),
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Invalid event data: {e}")

        if events:
            # 既存イベントとマージ
            existing = self.calendar.load_events()
            existing_ids = {e.event_id for e in existing}

            for event in events:
                if event.event_id not in existing_ids:
                    existing.append(event)

            self.calendar.save_events(existing)

        return len(events)

    def create_sample_calendar(self) -> None:
        """サンプルカレンダーを作成（テスト用）"""
        now = datetime.now()

        sample_events = [
            EconomicEvent(
                event_id="sample_1",
                title="米国雇用統計 (Non-Farm Payrolls)",
                country="US",
                currency="USD",
                timestamp=now + timedelta(days=7, hours=21, minutes=30),
                impact=EventImpact.HIGH,
                forecast="200K",
                previous="180K",
            ),
            EconomicEvent(
                event_id="sample_2",
                title="FOMC 金利決定",
                country="US",
                currency="USD",
                timestamp=now + timedelta(days=14, hours=3),
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                event_id="sample_3",
                title="ECB 金利決定",
                country="EU",
                currency="EUR",
                timestamp=now + timedelta(days=10, hours=20, minutes=45),
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                event_id="sample_4",
                title="日銀金融政策決定会合",
                country="JP",
                currency="JPY",
                timestamp=now + timedelta(days=5, hours=12),
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                event_id="sample_5",
                title="米国CPI (消費者物価指数)",
                country="US",
                currency="USD",
                timestamp=now + timedelta(days=3, hours=21, minutes=30),
                impact=EventImpact.HIGH,
                forecast="3.2%",
                previous="3.4%",
            ),
            EconomicEvent(
                event_id="sample_6",
                title="ユーロ圏GDP速報",
                country="EU",
                currency="EUR",
                timestamp=now + timedelta(days=8, hours=18),
                impact=EventImpact.MEDIUM,
            ),
        ]

        self.calendar.save_events(sample_events)
        logger.info(f"Created sample calendar with {len(sample_events)} events")

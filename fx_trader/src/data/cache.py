"""
ローカルデータキャッシュモジュール
OHLCVデータのローカルキャッシュでAPI負荷軽減・高速化
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVCache:
    """OHLCVデータキャッシュ（SQLite使用）"""

    def __init__(
        self,
        db_path: str = "data/ohlcv_cache.db",
        max_age_hours: int = 24,
    ):
        """
        Args:
            db_path: データベースファイルパス
            max_age_hours: キャッシュ有効期間（時間）
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age = timedelta(hours=max_age_hours)

        self._init_db()

    def _init_db(self) -> None:
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    cached_at TEXT NOT NULL,
                    UNIQUE(symbol, interval, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_lookup
                ON ohlcv(symbol, interval, timestamp)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    last_update TEXT NOT NULL,
                    start_time INTEGER,
                    end_time INTEGER,
                    PRIMARY KEY(symbol, interval)
                )
            """)
            conn.commit()

    def get(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        キャッシュからデータを取得

        Args:
            symbol: 通貨ペア
            interval: 時間足
            start_time: 開始時刻
            end_time: 終了時刻

        Returns:
            DataFrame、またはキャッシュミス時はNone
        """
        # キャッシュ有効性チェック
        if not self._is_cache_valid(symbol, interval):
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv
                    WHERE symbol = ? AND interval = ?
                """
                params: List[Any] = [symbol, interval]

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(int(start_time.timestamp() * 1000))
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(int(end_time.timestamp() * 1000))

                query += " ORDER BY timestamp ASC"

                df = pd.read_sql_query(query, conn, params=params)

                if df.empty:
                    return None

                # タイムスタンプを変換
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)

                logger.debug(f"Cache hit: {symbol} {interval}, {len(df)} rows")
                return df

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def put(
        self,
        symbol: str,
        interval: str,
        df: pd.DataFrame,
    ) -> bool:
        """
        データをキャッシュに保存

        Args:
            symbol: 通貨ペア
            interval: 時間足
            df: OHLCVデータ

        Returns:
            成功フラグ
        """
        if df.empty:
            return False

        try:
            now = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                # DataFrameを準備
                cache_df = df.copy()

                # インデックスがタイムスタンプの場合
                if isinstance(cache_df.index, pd.DatetimeIndex):
                    cache_df["timestamp"] = cache_df.index.astype(int) // 10**6
                    cache_df = cache_df.reset_index(drop=True)
                elif "timestamp" in cache_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(cache_df["timestamp"]):
                        cache_df["timestamp"] = cache_df["timestamp"].astype(int) // 10**6

                cache_df["symbol"] = symbol
                cache_df["interval"] = interval
                cache_df["cached_at"] = now

                # 必要なカラムのみ
                columns = ["symbol", "interval", "timestamp", "open", "high", "low", "close", "volume", "cached_at"]
                available_cols = [c for c in columns if c in cache_df.columns]

                if "volume" not in cache_df.columns:
                    cache_df["volume"] = 0

                # UPSERT
                for _, row in cache_df[available_cols].iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv
                        (symbol, interval, timestamp, open, high, low, close, volume, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["symbol"],
                        row["interval"],
                        int(row["timestamp"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row.get("volume", 0)),
                        row["cached_at"],
                    ))

                # メタデータ更新
                start_ts = int(cache_df["timestamp"].min())
                end_ts = int(cache_df["timestamp"].max())

                conn.execute("""
                    INSERT OR REPLACE INTO cache_meta
                    (symbol, interval, last_update, start_time, end_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, interval, now, start_ts, end_ts))

                conn.commit()

            logger.debug(f"Cache put: {symbol} {interval}, {len(df)} rows")
            return True

        except Exception as e:
            logger.error(f"Cache put error: {e}")
            return False

    def _is_cache_valid(self, symbol: str, interval: str) -> bool:
        """キャッシュが有効か確認"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT last_update FROM cache_meta
                    WHERE symbol = ? AND interval = ?
                """, (symbol, interval))

                row = cursor.fetchone()
                if not row:
                    return False

                last_update = datetime.fromisoformat(row[0])
                age = datetime.now() - last_update

                # インターバルによって有効期間を調整
                if interval in ["1m", "5m"]:
                    max_age = timedelta(minutes=5)
                elif interval in ["15m", "30m"]:
                    max_age = timedelta(minutes=15)
                elif interval in ["1h", "4h"]:
                    max_age = timedelta(hours=1)
                else:
                    max_age = self.max_age

                return age < max_age

        except Exception:
            return False

    def invalidate(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> None:
        """
        キャッシュを無効化

        Args:
            symbol: 通貨ペア（指定なしで全て）
            interval: 時間足（指定なしで全て）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol and interval:
                    conn.execute(
                        "DELETE FROM ohlcv WHERE symbol = ? AND interval = ?",
                        (symbol, interval)
                    )
                    conn.execute(
                        "DELETE FROM cache_meta WHERE symbol = ? AND interval = ?",
                        (symbol, interval)
                    )
                elif symbol:
                    conn.execute("DELETE FROM ohlcv WHERE symbol = ?", (symbol,))
                    conn.execute("DELETE FROM cache_meta WHERE symbol = ?", (symbol,))
                elif interval:
                    conn.execute("DELETE FROM ohlcv WHERE interval = ?", (interval,))
                    conn.execute("DELETE FROM cache_meta WHERE interval = ?", (interval,))
                else:
                    conn.execute("DELETE FROM ohlcv")
                    conn.execute("DELETE FROM cache_meta")

                conn.commit()
                logger.info(f"Cache invalidated: symbol={symbol}, interval={interval}")

        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 総レコード数
                cursor = conn.execute("SELECT COUNT(*) FROM ohlcv")
                total_rows = cursor.fetchone()[0]

                # シンボル別統計
                cursor = conn.execute("""
                    SELECT symbol, interval, COUNT(*) as count,
                           MIN(timestamp) as start, MAX(timestamp) as end
                    FROM ohlcv
                    GROUP BY symbol, interval
                """)

                symbols = {}
                for row in cursor.fetchall():
                    key = f"{row[0]}_{row[1]}"
                    symbols[key] = {
                        "count": row[2],
                        "start": datetime.fromtimestamp(row[3] / 1000).isoformat() if row[3] else None,
                        "end": datetime.fromtimestamp(row[4] / 1000).isoformat() if row[4] else None,
                    }

                # DBサイズ
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    "total_rows": total_rows,
                    "symbols": symbols,
                    "db_size_mb": round(db_size / (1024 * 1024), 2),
                }

        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {}

    def cleanup_old(self, days: int = 30) -> int:
        """
        古いデータを削除

        Args:
            days: 保持する日数

        Returns:
            削除したレコード数
        """
        try:
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_ts = int(cutoff.timestamp() * 1000)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM ohlcv WHERE timestamp < ?",
                    (cutoff_ts,)
                )
                deleted = cursor.rowcount
                conn.commit()

                # VACUUM
                conn.execute("VACUUM")

            logger.info(f"Cleaned up {deleted} old cache records")
            return deleted

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0


class CachedDataFetcher:
    """キャッシュ付きデータフェッチャー"""

    def __init__(
        self,
        fetcher,
        cache: Optional[OHLCVCache] = None,
    ):
        """
        Args:
            fetcher: 元のDataFetcher
            cache: OHLCVCacheインスタンス
        """
        self.fetcher = fetcher
        self.cache = cache or OHLCVCache()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        days: int = 7,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得（キャッシュ優先）

        Args:
            symbol: 通貨ペア
            interval: 時間足
            days: 取得日数
            use_cache: キャッシュを使用するか

        Returns:
            OHLCVデータ
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # キャッシュをチェック
        if use_cache:
            cached_df = self.cache.get(symbol, interval, start_time, end_time)
            if cached_df is not None and len(cached_df) > 0:
                logger.debug(f"Using cached data for {symbol} {interval}")
                return cached_df

        # APIから取得
        logger.debug(f"Fetching from API: {symbol} {interval}")
        df = self.fetcher.fetch_ohlcv(symbol, interval, days)

        # キャッシュに保存
        if not df.empty and use_cache:
            self.cache.put(symbol, interval, df)

        return df

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """ティッカーを取得（キャッシュなし）"""
        return self.fetcher.fetch_ticker(symbol)

    def calculate_spread(self, symbol: str) -> Tuple[float, bool]:
        """スプレッドを計算（キャッシュなし）"""
        return self.fetcher.calculate_spread(symbol)

    def invalidate_cache(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> None:
        """キャッシュを無効化"""
        self.cache.invalidate(symbol, interval)

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        return self.cache.get_stats()


class TickerCache:
    """ティッカーキャッシュ（メモリベース）"""

    def __init__(self, ttl_seconds: int = 5):
        """
        Args:
            ttl_seconds: キャッシュ有効期間（秒）
        """
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        """キャッシュからティッカーを取得"""
        if symbol not in self._cache:
            return None

        data, cached_at = self._cache[symbol]
        if datetime.now() - cached_at > self.ttl:
            del self._cache[symbol]
            return None

        return data

    def put(self, symbol: str, ticker: Dict[str, Any]) -> None:
        """ティッカーをキャッシュに保存"""
        self._cache[symbol] = (ticker, datetime.now())

    def invalidate(self, symbol: Optional[str] = None) -> None:
        """キャッシュを無効化"""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()

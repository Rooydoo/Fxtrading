"""
データ取得・整形モジュール
GMO FX APIからOHLCVデータを取得し、DataFrameに変換
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .gmo_client import GMOForexClient

logger = logging.getLogger(__name__)


class DataFetcher:
    """FXデータ取得・整形クラス"""

    # 時間足マッピング
    INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "10m": "10min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "4h": "4hour",
        "8h": "8hour",
        "12h": "12hour",
        "1d": "1day",
        "1w": "1week",
        "1M": "1month",
    }

    def __init__(self, client: Optional[GMOForexClient] = None):
        """
        Args:
            client: GMO APIクライアント (省略時は新規作成)
        """
        self.client = client or GMOForexClient()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        OHLCVデータを取得

        Args:
            symbol: 通貨ペア (例: EUR_USD, USD_JPY)
            interval: 時間足 (1m, 5m, 15m, 1h, 4h, 1d など)
            start_date: 開始日 (YYYYMMDD形式)
            end_date: 終了日 (YYYYMMDD形式)
            days: 取得日数 (start_date省略時に使用)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        gmo_interval = self.INTERVAL_MAP.get(interval, interval)

        # 日付範囲を決定
        if end_date is None:
            end_dt = datetime.now()
        else:
            end_dt = datetime.strptime(end_date, "%Y%m%d")

        if start_date is None:
            start_dt = end_dt - timedelta(days=days)
        else:
            start_dt = datetime.strptime(start_date, "%Y%m%d")

        all_data = []
        current_dt = start_dt

        logger.info(f"Fetching {symbol} {interval} data from {start_dt.date()} to {end_dt.date()}")

        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            try:
                response = self.client.get_klines(
                    symbol=symbol,
                    interval=gmo_interval,
                    date=date_str,
                )

                if response.get("status") == 0 and response.get("data"):
                    all_data.extend(response["data"])
                    logger.debug(f"Fetched {len(response['data'])} candles for {date_str}")

            except Exception as e:
                logger.warning(f"Failed to fetch data for {date_str}: {e}")

            current_dt += timedelta(days=1)

        if not all_data:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()

        df = self._parse_klines(all_data)
        return df

    def _parse_klines(self, data: List[Dict]) -> pd.DataFrame:
        """
        APIレスポンスをDataFrameに変換

        Args:
            data: APIレスポンスのdata部分

        Returns:
            整形されたDataFrame
        """
        records = []
        for item in data:
            records.append({
                "timestamp": pd.to_datetime(item["openTime"]),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
            })

        df = pd.DataFrame(records)

        if df.empty:
            return df

        # インデックス設定
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # 重複削除
        df = df[~df.index.duplicated(keep="last")]

        return df

    def fetch_ticker(self, symbol: str) -> Dict:
        """
        最新ティッカー取得

        Args:
            symbol: 通貨ペア

        Returns:
            ティッカー情報 (bid, ask, high, low, volume など)
        """
        response = self.client.get_ticker(symbol)

        if response.get("status") != 0:
            raise Exception(f"API error: {response}")

        data = response.get("data", [])
        if not data:
            raise Exception(f"No ticker data for {symbol}")

        ticker = data[0]
        return {
            "symbol": ticker["symbol"],
            "bid": float(ticker["bid"]),
            "ask": float(ticker["ask"]),
            "high": float(ticker["high"]),
            "low": float(ticker["low"]),
            "timestamp": pd.to_datetime(ticker["timestamp"]),
            "spread": float(ticker["ask"]) - float(ticker["bid"]),
        }

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ["15m", "1h"],
        days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """
        複数時間軸のデータを取得

        Args:
            symbol: 通貨ペア
            timeframes: 取得する時間軸リスト
            days: 取得日数

        Returns:
            時間軸をキーとしたDataFrame辞書
        """
        result = {}
        for tf in timeframes:
            logger.info(f"Fetching {tf} data for {symbol}")
            df = self.fetch_ohlcv(symbol, interval=tf, days=days)
            result[tf] = df
        return result

    def resample_ohlcv(
        self,
        df: pd.DataFrame,
        target_interval: str,
    ) -> pd.DataFrame:
        """
        OHLCVデータをリサンプリング

        Args:
            df: 元のOHLCVデータ
            target_interval: 目標時間軸 (15T, 1H, 4H, 1D など)

        Returns:
            リサンプリングされたDataFrame
        """
        if df.empty:
            return df

        resampled = df.resample(target_interval).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        })

        # NaN行を削除
        resampled.dropna(inplace=True)

        return resampled

    def calculate_spread(self, symbol: str) -> Tuple[float, bool]:
        """
        現在のスプレッドを計算し、異常かどうか判定

        Args:
            symbol: 通貨ペア

        Returns:
            (スプレッド値, 異常フラグ)
        """
        # 通常スプレッド (pips)
        normal_spreads = {
            "EUR_USD": 1.5,
            "USD_JPY": 1.5,
            "GBP_USD": 2.0,
            "AUD_USD": 2.0,
        }

        ticker = self.fetch_ticker(symbol)
        current_spread = ticker["spread"]

        # pip値に変換
        if "JPY" in symbol:
            spread_pips = current_spread * 100  # 0.01 = 1 pip
        else:
            spread_pips = current_spread * 10000  # 0.0001 = 1 pip

        normal = normal_spreads.get(symbol, 2.0)
        is_abnormal = spread_pips > normal * 2  # 通常の2倍以上で異常

        return spread_pips, is_abnormal


class PaperDataFetcher(DataFetcher):
    """ペーパートレード用データフェッチャー（キャッシュ機能付き）"""

    def __init__(self, client: Optional[GMOForexClient] = None):
        super().__init__(client)
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """キャッシュを活用したOHLCV取得"""
        cache_key = f"{symbol}_{interval}_{start_date}_{end_date}_{days}"

        if cache_key in self._cache:
            logger.debug(f"Using cached data for {cache_key}")
            return self._cache[cache_key].copy()

        df = super().fetch_ohlcv(symbol, interval, start_date, end_date, days)
        self._cache[cache_key] = df.copy()

        return df

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        logger.info("Data cache cleared")

"""
特徴量生成モジュール
テクニカル指標や時間特徴量を計算
"""
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """特徴量生成クラス"""

    def __init__(self, config_path: str = "config/features.yaml"):
        """
        Args:
            config_path: 特徴量設定ファイルのパス
        """
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """デフォルト設定"""
        return {
            "price_features": {"enabled": True},
            "technical_indicators": {"enabled": True},
            "time_features": {"enabled": True},
            "lag_features": {"enabled": True, "lags": [1, 2, 3, 5]},
        }

    def build_all_features(
        self,
        df: pd.DataFrame,
        higher_tf_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        全ての特徴量を生成

        Args:
            df: OHLCVデータ
            higher_tf_df: 上位時間軸データ (オプション)

        Returns:
            特徴量付きDataFrame
        """
        if df.empty:
            return df

        result = df.copy()

        # 価格系特徴量
        if self.config.get("price_features", {}).get("enabled", True):
            result = self._add_price_features(result)

        # テクニカル指標
        if self.config.get("technical_indicators", {}).get("enabled", True):
            result = self._add_technical_indicators(result)

        # 時間特徴量
        if self.config.get("time_features", {}).get("enabled", True):
            result = self._add_time_features(result)

        # 上位時間軸特徴量
        if higher_tf_df is not None and self.config.get("higher_timeframe", {}).get("enabled", True):
            result = self._add_higher_tf_features(result, higher_tf_df)

        # ラグ特徴量
        if self.config.get("lag_features", {}).get("enabled", True):
            result = self._add_lag_features(result)

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格系特徴量を追加"""
        config = self.config.get("price_features", {}).get("features", {})

        # リターン
        if config.get("returns", True):
            for period in [1, 2, 3, 5, 10]:
                df[f"return_{period}"] = df["close"].pct_change(period)

        # 対数リターン
        if config.get("log_returns", True):
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # ボラティリティ (リターンの標準偏差)
        if config.get("volatility", True):
            df["volatility_10"] = df["log_return"].rolling(10).std()
            df["volatility_20"] = df["log_return"].rolling(20).std()

        # レンジ
        if config.get("range", True):
            df["range"] = df["high"] - df["low"]
            df["range_pct"] = df["range"] / df["close"]

        # ローソク足の実体比率
        if config.get("body_ratio", True):
            body = abs(df["close"] - df["open"])
            df["body_ratio"] = body / df["range"].replace(0, np.nan)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を追加"""
        config = self.config.get("technical_indicators", {})

        # RSI
        rsi_config = config.get("rsi", {})
        if rsi_config.get("enabled", True):
            for period in rsi_config.get("periods", [7, 14, 21]):
                df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # MACD
        macd_config = config.get("macd", {})
        if macd_config.get("enabled", True):
            fast = macd_config.get("fast_period", 12)
            slow = macd_config.get("slow_period", 26)
            signal = macd_config.get("signal_period", 9)
            df["macd"], df["macd_signal"], df["macd_histogram"] = self._calculate_macd(
                df["close"], fast, slow, signal
            )

        # ボリンジャーバンド
        bb_config = config.get("bollinger_bands", {})
        if bb_config.get("enabled", True):
            period = bb_config.get("period", 20)
            std_dev = bb_config.get("std_dev", 2.0)
            df["bb_middle"] = df["close"].rolling(period).mean()
            rolling_std = df["close"].rolling(period).std()
            df["bb_upper"] = df["bb_middle"] + std_dev * rolling_std
            df["bb_lower"] = df["bb_middle"] - std_dev * rolling_std
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            df["bb_position"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"]
            ).replace(0, np.nan)

        # ATR
        atr_config = config.get("atr", {})
        if atr_config.get("enabled", True):
            for period in atr_config.get("periods", [7, 14, 21]):
                df[f"atr_{period}"] = self._calculate_atr(df, period)

        # ADX
        adx_config = config.get("adx", {})
        if adx_config.get("enabled", True):
            period = adx_config.get("period", 14)
            df["adx"] = self._calculate_adx(df, period)

        # 移動平均
        ma_config = config.get("moving_averages", {})
        if ma_config.get("enabled", True):
            for period in ma_config.get("sma_periods", [5, 10, 20, 50, 100]):
                df[f"sma_{period}"] = df["close"].rolling(period).mean()
            for period in ma_config.get("ema_periods", [5, 10, 20, 50]):
                df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

            # 価格とMAの距離
            df["price_sma20_distance"] = (df["close"] - df["sma_20"]) / df["sma_20"]

            # MAクロスオーバー
            df["ma_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)

        # ストキャスティクス
        stoch_config = config.get("stochastic", {})
        if stoch_config.get("enabled", True):
            k_period = stoch_config.get("k_period", 14)
            d_period = stoch_config.get("d_period", 3)
            df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(df, k_period, d_period)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徴量を追加"""
        config = self.config.get("time_features", {})
        features = config.get("features", {})

        if features.get("hour", True):
            df["hour"] = df.index.hour

        if features.get("day_of_week", True):
            df["day_of_week"] = df.index.dayofweek

        if features.get("day_of_month", True):
            df["day_of_month"] = df.index.day

        if features.get("month", True):
            df["month"] = df.index.month

        if features.get("is_month_end", True):
            df["is_month_end"] = df.index.is_month_end.astype(int)

        # セッション分類
        sessions_config = config.get("sessions", {})
        if sessions_config.get("enabled", True):
            hour_utc = df.index.hour  # UTCを想定

            # 東京セッション (00:00-09:00 UTC = 09:00-18:00 JST)
            tokyo = sessions_config.get("tokyo", [0, 9])
            df["session_tokyo"] = ((hour_utc >= tokyo[0]) & (hour_utc < tokyo[1])).astype(int)

            # ロンドンセッション (08:00-17:00 UTC)
            london = sessions_config.get("london", [8, 17])
            df["session_london"] = ((hour_utc >= london[0]) & (hour_utc < london[1])).astype(int)

            # NYセッション (13:00-22:00 UTC)
            ny = sessions_config.get("new_york", [13, 22])
            df["session_ny"] = ((hour_utc >= ny[0]) & (hour_utc < ny[1])).astype(int)

        return df

    def _add_higher_tf_features(
        self,
        df: pd.DataFrame,
        higher_tf_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """上位時間軸特徴量を追加"""
        config = self.config.get("higher_timeframe", {}).get("features", {})

        # 上位時間軸のRSI
        if config.get("rsi", True):
            htf_rsi = self._calculate_rsi(higher_tf_df["close"], 14)
            htf_rsi.name = "htf_rsi"
            df = self._merge_higher_tf(df, htf_rsi)

        # 上位時間軸のトレンド方向
        if config.get("trend", True):
            htf_sma20 = higher_tf_df["close"].rolling(20).mean()
            htf_sma50 = higher_tf_df["close"].rolling(50).mean()
            htf_trend = (htf_sma20 > htf_sma50).astype(int)
            htf_trend.name = "htf_trend"
            df = self._merge_higher_tf(df, htf_trend)

        # 上位時間軸のボラティリティ
        if config.get("volatility", True):
            htf_vol = higher_tf_df["close"].pct_change().rolling(20).std()
            htf_vol.name = "htf_volatility"
            df = self._merge_higher_tf(df, htf_vol)

        return df

    def _merge_higher_tf(
        self,
        df: pd.DataFrame,
        series: pd.Series,
    ) -> pd.DataFrame:
        """上位時間軸データをマージ"""
        # リサンプルして前方補完
        series_reindexed = series.reindex(df.index, method="ffill")
        df[series.name] = series_reindexed
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラグ特徴量を追加"""
        config = self.config.get("lag_features", {})
        lags = config.get("lags", [1, 2, 3, 5])
        apply_to = config.get("apply_to", ["return_1", "rsi_14", "macd_histogram"])

        for col in apply_to:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

        return df

    # ==================== テクニカル指標計算 ====================

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR計算"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX計算"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self._calculate_atr(df, period)

        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_stochastic(
        self,
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple:
        """ストキャスティクス計算"""
        low_min = df["low"].rolling(k_period).min()
        high_max = df["high"].rolling(k_period).max()

        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        stoch_d = stoch_k.rolling(d_period).mean()

        return stoch_k, stoch_d

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        特徴量名のリストを取得

        Args:
            df: 特徴量付きDataFrame

        Returns:
            特徴量名リスト (OHLC以外)
        """
        exclude = ["open", "high", "low", "close", "volume"]
        return [col for col in df.columns if col not in exclude]

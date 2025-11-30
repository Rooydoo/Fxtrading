"""
特徴量選択モジュール
特徴量の重要度分析、選択、前処理を行う
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest

logger = logging.getLogger(__name__)


class FeatureSelector:
    """特徴量選択・前処理クラス"""

    def __init__(
        self,
        scaling_method: str = "standard",
        exclude_from_scaling: Optional[List[str]] = None,
    ):
        """
        Args:
            scaling_method: スケーリング方法 (standard, minmax, robust)
            exclude_from_scaling: スケーリングから除外する特徴量
        """
        self.scaling_method = scaling_method
        self.exclude_from_scaling = exclude_from_scaling or []
        self.scaler = None
        self.selected_features: List[str] = []
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target: Optional[pd.Series] = None,
    ) -> "FeatureSelector":
        """
        スケーラーと特徴量選択をフィット

        Args:
            df: 特徴量DataFrame
            feature_columns: 使用する特徴量カラム
            target: ターゲット変数 (特徴量選択に使用)
        """
        # スケーリング対象の特徴量を特定
        scale_columns = [
            col for col in feature_columns
            if col not in self.exclude_from_scaling
            and not any(excl in col for excl in self.exclude_from_scaling)
        ]

        # スケーラーの選択と学習
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        if scale_columns:
            # NaN除去してフィット
            valid_data = df[scale_columns].dropna()
            if not valid_data.empty:
                self.scaler.fit(valid_data)

        self.selected_features = feature_columns
        self._scale_columns = scale_columns
        self._fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量を変換

        Args:
            df: 特徴量DataFrame

        Returns:
            変換後のDataFrame
        """
        if not self._fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform")

        result = df.copy()

        # スケーリング
        if self.scaler and self._scale_columns:
            cols_to_scale = [c for c in self._scale_columns if c in result.columns]
            if cols_to_scale:
                result[cols_to_scale] = self.scaler.transform(result[cols_to_scale])

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """フィットと変換を同時実行"""
        self.fit(df, feature_columns, target)
        return self.transform(df)

    def select_by_importance(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        feature_columns: List[str],
        method: str = "mutual_info",
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        重要度による特徴量選択

        Args:
            df: 特徴量DataFrame
            target: ターゲット変数
            feature_columns: 候補特徴量
            method: 選択方法 (mutual_info, correlation)
            top_k: 上位K個を選択
            threshold: 重要度の閾値

        Returns:
            (選択された特徴量リスト, 重要度辞書)
        """
        # 欠損値のある行を除去
        valid_mask = df[feature_columns].notna().all(axis=1) & target.notna()
        X = df.loc[valid_mask, feature_columns]
        y = target.loc[valid_mask]

        if len(X) < 100:
            logger.warning(f"Not enough samples for feature selection: {len(X)}")
            return feature_columns, {col: 1.0 for col in feature_columns}

        importance = {}

        if method == "mutual_info":
            # 相互情報量
            mi_scores = mutual_info_classif(X, y, random_state=42)
            for col, score in zip(feature_columns, mi_scores):
                importance[col] = score

        elif method == "correlation":
            # ターゲットとの相関
            for col in feature_columns:
                corr = abs(df[col].corr(target))
                importance[col] = corr if not np.isnan(corr) else 0

        # ソート
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # 選択
        selected = []
        if top_k:
            selected = [f[0] for f in sorted_features[:top_k]]
        elif threshold:
            selected = [f[0] for f, score in sorted_features if score >= threshold]
        else:
            selected = [f[0] for f in sorted_features]

        self.selected_features = selected
        return selected, importance

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        threshold: float = 0.95,
    ) -> List[str]:
        """
        高相関特徴量を除去

        Args:
            df: 特徴量DataFrame
            feature_columns: 候補特徴量
            threshold: 相関の閾値

        Returns:
            除去後の特徴量リスト
        """
        corr_matrix = df[feature_columns].corr().abs()

        # 上三角行列
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 閾値以上の相関を持つ特徴量を特定
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]

        selected = [col for col in feature_columns if col not in to_drop]
        logger.info(f"Removed {len(to_drop)} highly correlated features")

        return selected

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        欠損値処理

        Args:
            df: DataFrame
            method: 処理方法 (forward_fill, drop, mean, median)

        Returns:
            処理後のDataFrame
        """
        result = df.copy()

        if method == "forward_fill":
            result = result.ffill()
        elif method == "drop":
            result = result.dropna()
        elif method == "mean":
            result = result.fillna(result.mean())
        elif method == "median":
            result = result.fillna(result.median())
        else:
            raise ValueError(f"Unknown method: {method}")

        return result

    def handle_outliers(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        method: str = "clip",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        外れ値処理

        Args:
            df: DataFrame
            feature_columns: 処理対象カラム
            method: 処理方法 (clip, remove)
            threshold: 標準偏差の倍数

        Returns:
            処理後のDataFrame
        """
        result = df.copy()

        for col in feature_columns:
            if col not in result.columns:
                continue

            mean = result[col].mean()
            std = result[col].std()

            if method == "clip":
                lower = mean - threshold * std
                upper = mean + threshold * std
                result[col] = result[col].clip(lower, upper)
            elif method == "remove":
                mask = (result[col] >= mean - threshold * std) & (
                    result[col] <= mean + threshold * std
                )
                result.loc[~mask, col] = np.nan

        return result

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        handle_missing: str = "forward_fill",
        handle_outliers_method: Optional[str] = "clip",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        特徴量の前処理パイプライン

        Args:
            df: 生データ
            feature_columns: 使用する特徴量 (省略時は自動検出)
            handle_missing: 欠損値処理方法
            handle_outliers_method: 外れ値処理方法 (Noneでスキップ)

        Returns:
            (処理済みDataFrame, 特徴量リスト)
        """
        # 特徴量カラムの特定
        if feature_columns is None:
            exclude = ["open", "high", "low", "close", "volume", "target"]
            feature_columns = [col for col in df.columns if col not in exclude]

        result = df.copy()

        # 欠損値処理
        result = self.handle_missing_values(result, method=handle_missing)

        # 外れ値処理
        if handle_outliers_method:
            numeric_cols = [
                col for col in feature_columns
                if col in result.columns and result[col].dtype in ["float64", "float32", "int64", "int32"]
            ]
            result = self.handle_outliers(
                result, numeric_cols, method=handle_outliers_method
            )

        # 無限値の処理
        result = result.replace([np.inf, -np.inf], np.nan)
        result = self.handle_missing_values(result, method=handle_missing)

        return result, feature_columns


class FeatureDriftDetector:
    """特徴量ドリフト検出"""

    def __init__(self, baseline_df: pd.DataFrame, feature_columns: List[str]):
        """
        Args:
            baseline_df: ベースラインデータ
            feature_columns: モニタリング対象特徴量
        """
        self.baseline_stats = self._compute_stats(baseline_df, feature_columns)
        self.feature_columns = feature_columns

    def _compute_stats(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """統計量を計算"""
        stats = {}
        for col in feature_columns:
            if col in df.columns:
                stats[col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "q25": df[col].quantile(0.25),
                    "q75": df[col].quantile(0.75),
                }
        return stats

    def calculate_psi(
        self,
        current_df: pd.DataFrame,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Population Stability Index (PSI) を計算

        Args:
            current_df: 現在のデータ
            n_bins: ビン数

        Returns:
            各特徴量のPSI値
        """
        psi_scores = {}

        for col in self.feature_columns:
            if col not in current_df.columns:
                continue

            baseline_data = self.baseline_stats.get(col)
            if baseline_data is None:
                continue

            # ビン分割
            current_values = current_df[col].dropna()
            if len(current_values) < n_bins:
                continue

            try:
                # 同じビン境界を使用
                bins = np.linspace(baseline_data["min"], baseline_data["max"], n_bins + 1)

                # ベースラインの期待分布 (均等と仮定)
                expected = np.ones(n_bins) / n_bins

                # 現在の分布
                actual, _ = np.histogram(current_values, bins=bins)
                actual = actual / actual.sum()

                # PSI計算
                # ゼロ除算を避けるため、小さな値を追加
                expected = np.maximum(expected, 1e-10)
                actual = np.maximum(actual, 1e-10)

                psi = np.sum((actual - expected) * np.log(actual / expected))
                psi_scores[col] = psi

            except Exception as e:
                logger.warning(f"Failed to calculate PSI for {col}: {e}")

        return psi_scores

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        psi_threshold: float = 0.25,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        ドリフトを検出

        Args:
            current_df: 現在のデータ
            psi_threshold: PSI閾値

        Returns:
            (ドリフト検出フラグ, PSIスコア辞書)
        """
        psi_scores = self.calculate_psi(current_df)

        drifted_features = [
            col for col, psi in psi_scores.items()
            if psi > psi_threshold
        ]

        is_drifted = len(drifted_features) > 0

        if is_drifted:
            logger.warning(f"Feature drift detected: {drifted_features}")

        return is_drifted, psi_scores

"""
特徴量ドリフト検出モジュール
データ分布の変化を検出
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """特徴量ドリフト検出クラス"""

    def __init__(
        self,
        features_to_monitor: List[str],
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.05,
    ):
        """
        Args:
            features_to_monitor: モニタリング対象特徴量
            psi_threshold: PSI閾値
            ks_threshold: KSテストのp値閾値
        """
        self.features_to_monitor = features_to_monitor
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

        self.baseline_stats: Dict[str, Dict[str, Any]] = {}
        self._baseline_fitted = False

    def fit_baseline(self, df: pd.DataFrame) -> None:
        """
        ベースライン統計を計算

        Args:
            df: ベースラインデータ
        """
        for feature in self.features_to_monitor:
            if feature not in df.columns:
                continue

            data = df[feature].dropna()
            if len(data) < 10:
                continue

            self.baseline_stats[feature] = {
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "q25": data.quantile(0.25),
                "q50": data.quantile(0.50),
                "q75": data.quantile(0.75),
                "data": data.values,  # PSI計算用
            }

        self._baseline_fitted = True
        logger.info(f"Baseline fitted for {len(self.baseline_stats)} features")

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Population Stability Index (PSI) を計算

        Args:
            baseline: ベースラインデータ
            current: 現在のデータ
            n_bins: ビン数

        Returns:
            PSI値
        """
        # 共通のビン境界を作成
        combined = np.concatenate([baseline, current])
        _, bin_edges = np.histogram(combined, bins=n_bins)

        # 各ビンの比率を計算
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # 比率に変換 (ゼロ除算回避)
        baseline_pct = (baseline_counts + 1) / (len(baseline) + n_bins)
        current_pct = (current_counts + 1) / (len(current) + n_bins)

        # PSI計算
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi

    def ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov検定

        Args:
            baseline: ベースラインデータ
            current: 現在のデータ

        Returns:
            (統計量, p値)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        return statistic, p_value

    def detect_drift(
        self,
        df: pd.DataFrame,
        method: str = "psi",
    ) -> Dict[str, Any]:
        """
        ドリフトを検出

        Args:
            df: 現在のデータ
            method: 検出方法 (psi, ks_test, both)

        Returns:
            検出結果
        """
        if not self._baseline_fitted:
            raise RuntimeError("Baseline not fitted. Call fit_baseline first.")

        results = {
            "drifted_features": [],
            "feature_scores": {},
            "overall_drift": False,
            "method": method,
        }

        for feature in self.features_to_monitor:
            if feature not in df.columns or feature not in self.baseline_stats:
                continue

            current_data = df[feature].dropna().values
            baseline_data = self.baseline_stats[feature]["data"]

            if len(current_data) < 10:
                continue

            feature_result = {"feature": feature, "drift_detected": False}

            if method in ("psi", "both"):
                psi = self.calculate_psi(baseline_data, current_data)
                feature_result["psi"] = psi
                if psi > self.psi_threshold:
                    feature_result["drift_detected"] = True

            if method in ("ks_test", "both"):
                ks_stat, p_value = self.ks_test(baseline_data, current_data)
                feature_result["ks_statistic"] = ks_stat
                feature_result["ks_pvalue"] = p_value
                if p_value < self.ks_threshold:
                    feature_result["drift_detected"] = True

            results["feature_scores"][feature] = feature_result

            if feature_result["drift_detected"]:
                results["drifted_features"].append(feature)

        results["overall_drift"] = len(results["drifted_features"]) > 0
        results["drift_rate"] = len(results["drifted_features"]) / len(self.features_to_monitor)

        if results["overall_drift"]:
            logger.warning(f"Drift detected in features: {results['drifted_features']}")

        return results

    def get_distribution_comparison(
        self,
        df: pd.DataFrame,
        feature: str,
    ) -> Dict[str, Any]:
        """
        特徴量の分布比較を取得

        Args:
            df: 現在のデータ
            feature: 特徴量名

        Returns:
            分布比較情報
        """
        if feature not in self.baseline_stats:
            return {}

        baseline = self.baseline_stats[feature]
        current_data = df[feature].dropna()

        return {
            "feature": feature,
            "baseline": {
                "mean": baseline["mean"],
                "std": baseline["std"],
                "min": baseline["min"],
                "max": baseline["max"],
                "q25": baseline["q25"],
                "q50": baseline["q50"],
                "q75": baseline["q75"],
            },
            "current": {
                "mean": current_data.mean(),
                "std": current_data.std(),
                "min": current_data.min(),
                "max": current_data.max(),
                "q25": current_data.quantile(0.25),
                "q50": current_data.quantile(0.50),
                "q75": current_data.quantile(0.75),
            },
            "mean_shift": (current_data.mean() - baseline["mean"]) / (baseline["std"] + 1e-10),
            "std_ratio": current_data.std() / (baseline["std"] + 1e-10),
        }


class PredictionDriftDetector:
    """予測ドリフト検出"""

    def __init__(
        self,
        accuracy_threshold: float = 0.15,
        window_size: int = 100,
    ):
        """
        Args:
            accuracy_threshold: 精度乖離閾値
            window_size: ウィンドウサイズ
        """
        self.accuracy_threshold = accuracy_threshold
        self.window_size = window_size

        self.predictions: List[int] = []
        self.actuals: List[int] = []
        self.baseline_accuracy: Optional[float] = None

    def set_baseline_accuracy(self, accuracy: float) -> None:
        """ベースライン精度を設定"""
        self.baseline_accuracy = accuracy
        logger.info(f"Baseline accuracy set: {accuracy:.4f}")

    def record_prediction(self, prediction: int, actual: int) -> None:
        """
        予測を記録

        Args:
            prediction: 予測値
            actual: 実績値
        """
        self.predictions.append(prediction)
        self.actuals.append(actual)

        # ウィンドウサイズを超えたら古いものを削除
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)

    def detect_drift(self) -> Dict[str, Any]:
        """
        予測ドリフトを検出

        Returns:
            検出結果
        """
        if self.baseline_accuracy is None:
            return {"drift_detected": False, "error": "Baseline not set"}

        if len(self.predictions) < 10:
            return {"drift_detected": False, "error": "Not enough data"}

        # 現在の精度計算
        correct = sum(
            1 for p, a in zip(self.predictions, self.actuals)
            if p == a
        )
        current_accuracy = correct / len(self.predictions)

        # 乖離計算
        deviation = abs(current_accuracy - self.baseline_accuracy)
        drift_detected = deviation > self.accuracy_threshold

        result = {
            "drift_detected": drift_detected,
            "baseline_accuracy": self.baseline_accuracy,
            "current_accuracy": current_accuracy,
            "deviation": deviation,
            "threshold": self.accuracy_threshold,
            "sample_size": len(self.predictions),
        }

        if drift_detected:
            logger.warning(f"Prediction drift detected: {deviation:.4f} > {self.accuracy_threshold}")

        return result

    def reset(self) -> None:
        """記録をリセット"""
        self.predictions.clear()
        self.actuals.clear()

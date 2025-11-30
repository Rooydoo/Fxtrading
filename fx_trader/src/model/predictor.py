"""
LightGBM予測モジュール
シグナル生成と予測確信度の計算
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)


class SignalPredictor:
    """LightGBMベースのシグナル予測クラス"""

    # シグナル定義
    SIGNAL_LONG = 1
    SIGNAL_SHORT = -1
    SIGNAL_NEUTRAL = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold_long: float = 0.55,
        threshold_short: float = 0.55,
    ):
        """
        Args:
            model_path: 保存されたモデルのパス
            threshold_long: ロングシグナルの確信度閾値
            threshold_short: ショートシグナルの確信度閾値
        """
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.metadata: Dict[str, Any] = {}

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """
        モデルを読み込み

        Args:
            model_path: モデルファイルのパス
        """
        path = Path(model_path)

        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.feature_names = data.get("feature_names", [])
                self.metadata = data.get("metadata", {})
        else:
            self.model = lgb.Booster(model_file=str(path))
            self.feature_names = self.model.feature_name()

        logger.info(f"Model loaded from {model_path}")

    def save(self, model_path: str) -> None:
        """
        モデルを保存

        Args:
            model_path: 保存先パス
        """
        if self.model is None:
            raise RuntimeError("No model to save")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {model_path}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        確率予測

        Args:
            df: 特徴量DataFrame

        Returns:
            予測確率配列 (shape: n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 特徴量の選択と順序合わせ
        X = df[self.feature_names].values

        # 予測
        proba = self.model.predict(X)

        # バイナリ分類の場合
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])

        return proba

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        クラス予測

        Args:
            df: 特徴量DataFrame

        Returns:
            予測クラス配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        proba = self.predict_proba(df)
        return np.argmax(proba, axis=1)

    def generate_signal(
        self,
        df: pd.DataFrame,
        return_confidence: bool = True,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        シグナル生成

        Args:
            df: 特徴量DataFrame (最新1行分)
            return_confidence: 確信度を返すか

        Returns:
            (シグナル, 確信度, 詳細情報)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(df) == 0:
            return self.SIGNAL_NEUTRAL, 0.0, {}

        # 最新のデータで予測
        latest = df.iloc[[-1]]
        proba = self.predict_proba(latest)[0]

        # 3クラス分類 (0: Short, 1: Neutral, 2: Long) を想定
        # 2クラスの場合は (0: Down, 1: Up)
        if len(proba) == 3:
            prob_short = proba[0]
            prob_neutral = proba[1]
            prob_long = proba[2]
        else:
            # 2クラス分類
            prob_long = proba[1]
            prob_short = proba[0]
            prob_neutral = 0.0

        # シグナル判定
        signal = self.SIGNAL_NEUTRAL
        confidence = 0.0

        if prob_long > self.threshold_long and prob_long > prob_short:
            signal = self.SIGNAL_LONG
            confidence = prob_long
        elif prob_short > self.threshold_short and prob_short > prob_long:
            signal = self.SIGNAL_SHORT
            confidence = prob_short
        else:
            confidence = max(prob_long, prob_short)

        details = {
            "prob_long": float(prob_long),
            "prob_short": float(prob_short),
            "prob_neutral": float(prob_neutral),
            "threshold_long": self.threshold_long,
            "threshold_short": self.threshold_short,
            "timestamp": datetime.now().isoformat(),
        }

        return signal, confidence, details

    def get_feature_importance(
        self,
        importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            importance_type: 重要度タイプ (gain, split)

        Returns:
            特徴量名と重要度の辞書
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        importance = self.model.feature_importance(importance_type=importance_type)
        return dict(zip(self.feature_names, importance))

    def set_thresholds(
        self,
        threshold_long: float,
        threshold_short: float,
    ) -> None:
        """確信度閾値を設定"""
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short


class EnsemblePredictor:
    """複数モデルのアンサンブル予測"""

    def __init__(self, predictors: List[SignalPredictor]):
        """
        Args:
            predictors: SignalPredictorのリスト
        """
        self.predictors = predictors

    def generate_signal(
        self,
        df: pd.DataFrame,
        voting: str = "soft",
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        アンサンブルシグナル生成

        Args:
            df: 特徴量DataFrame
            voting: 投票方式 (soft: 確率平均, hard: 多数決)

        Returns:
            (シグナル, 確信度, 詳細情報)
        """
        signals = []
        confidences = []
        all_details = []

        for predictor in self.predictors:
            try:
                signal, conf, details = predictor.generate_signal(df)
                signals.append(signal)
                confidences.append(conf)
                all_details.append(details)
            except Exception as e:
                logger.warning(f"Predictor failed: {e}")

        if not signals:
            return SignalPredictor.SIGNAL_NEUTRAL, 0.0, {}

        if voting == "soft":
            # 確信度加重平均
            avg_conf = np.mean(confidences)
            # 加重投票
            weighted_signal = np.sum(np.array(signals) * np.array(confidences))
            if weighted_signal > 0:
                final_signal = SignalPredictor.SIGNAL_LONG
            elif weighted_signal < 0:
                final_signal = SignalPredictor.SIGNAL_SHORT
            else:
                final_signal = SignalPredictor.SIGNAL_NEUTRAL
        else:
            # ハード投票
            final_signal = int(np.sign(np.sum(signals)))
            avg_conf = np.mean(confidences)

        details = {
            "individual_signals": signals,
            "individual_confidences": confidences,
            "voting_method": voting,
            "n_predictors": len(self.predictors),
        }

        return final_signal, avg_conf, details


class PredictionLogger:
    """予測ログ管理"""

    def __init__(self, log_dir: str = "logs/predictions"):
        """
        Args:
            log_dir: ログ保存ディレクトリ
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions: List[Dict] = []

    def log_prediction(
        self,
        symbol: str,
        signal: int,
        confidence: float,
        details: Dict[str, Any],
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        予測をログに記録

        Args:
            symbol: 通貨ペア
            signal: シグナル
            confidence: 確信度
            details: 詳細情報
            features: 使用した特徴量値
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "signal_name": self._signal_name(signal),
            "confidence": confidence,
            "details": details,
        }

        if features:
            record["features"] = features

        self.predictions.append(record)
        logger.debug(f"Prediction logged: {symbol} {record['signal_name']} ({confidence:.2%})")

    def _signal_name(self, signal: int) -> str:
        """シグナルを文字列に変換"""
        if signal == SignalPredictor.SIGNAL_LONG:
            return "LONG"
        elif signal == SignalPredictor.SIGNAL_SHORT:
            return "SHORT"
        else:
            return "NEUTRAL"

    def save_logs(self, filename: Optional[str] = None) -> str:
        """
        ログをファイルに保存

        Args:
            filename: ファイル名 (省略時は日付ベース)

        Returns:
            保存先パス
        """
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"

        df = pd.DataFrame(self.predictions)
        path = self.log_dir / filename
        df.to_csv(path, index=False)

        logger.info(f"Predictions saved to {path}")
        return str(path)

    def get_recent_predictions(
        self,
        n: int = 100,
        symbol: Optional[str] = None,
    ) -> List[Dict]:
        """
        最近の予測を取得

        Args:
            n: 取得件数
            symbol: フィルタする通貨ペア

        Returns:
            予測リスト
        """
        predictions = self.predictions
        if symbol:
            predictions = [p for p in predictions if p["symbol"] == symbol]
        return predictions[-n:]

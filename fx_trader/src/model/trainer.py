"""
LightGBMモデル学習モジュール
ウォークフォワード検証、ハイパーパラメータチューニングを含む
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class ModelTrainer:
    """LightGBMモデル学習クラス"""

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
    ):
        """
        Args:
            params: LightGBMパラメータ
            early_stopping_rounds: 早期停止ラウンド数
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.training_history: List[Dict] = []

    def prepare_target(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        lookahead: int = 1,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        ターゲット変数を生成

        Args:
            df: OHLCVデータ
            target_col: 価格カラム
            lookahead: 予測期間
            threshold: 方向判定の閾値

        Returns:
            ターゲット変数 (1: 上昇, 0: 下降)
        """
        future_return = df[target_col].pct_change(lookahead).shift(-lookahead)

        # バイナリ分類
        target = (future_return > threshold).astype(int)

        return target

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
    ) -> lgb.Booster:
        """
        モデルを学習

        Args:
            X: 訓練特徴量
            y: 訓練ターゲット
            X_val: 検証特徴量
            y_val: 検証ターゲット
            feature_names: 特徴量名

        Returns:
            学習済みBooster
        """
        self.feature_names = feature_names or list(X.columns)

        # データセット作成
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # 学習
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(f"Model trained with {self.model.num_trees()} trees")

        return self.model

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target: pd.Series,
        n_splits: int = 4,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        ウォークフォワード検証

        Args:
            df: 特徴量DataFrame
            feature_columns: 使用する特徴量
            target: ターゲット変数
            n_splits: 分割数
            test_size: テストサイズ比率

        Returns:
            検証結果
        """
        # NaN除去
        valid_mask = df[feature_columns].notna().all(axis=1) & target.notna()
        X = df.loc[valid_mask, feature_columns]
        y = target.loc[valid_mask]

        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)

        results = {
            "scores": [],
            "predictions": [],
            "actuals": [],
            "fold_details": [],
        }

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            logger.info(f"Fold {fold + 1}/{n_splits}: Train={len(X_train)}, Test={len(X_test)}")

            # 学習
            model = self.train(
                X_train, y_train,
                X_test, y_test,
                feature_names=feature_columns,
            )

            # 予測
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # 評価
            accuracy = (y_pred == y_test).mean()
            results["scores"].append(accuracy)
            results["predictions"].extend(y_pred.tolist())
            results["actuals"].extend(y_test.tolist())

            results["fold_details"].append({
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": accuracy,
                "train_start": X_train.index.min(),
                "train_end": X_train.index.max(),
                "test_start": X_test.index.min(),
                "test_end": X_test.index.max(),
            })

        # 全体の評価
        results["mean_accuracy"] = np.mean(results["scores"])
        results["std_accuracy"] = np.std(results["scores"])

        # 方向精度
        preds = np.array(results["predictions"])
        actuals = np.array(results["actuals"])
        results["direction_accuracy"] = (preds == actuals).mean()

        logger.info(f"Walk-forward validation complete: "
                    f"Mean accuracy={results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")

        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "walk_forward",
            "results": results,
        })

        return results

    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optunaによるハイパーパラメータチューニング

        Args:
            X: 特徴量
            y: ターゲット
            n_trials: 試行回数
            timeout: タイムアウト秒数

        Returns:
            最適パラメータ
        """
        try:
            import optuna
            from optuna.integration import LightGBMPruningCallback
        except ImportError:
            logger.warning("Optuna not installed, skipping hyperparameter tuning")
            return self.params

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbose": -1,
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]

                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=30),
                        LightGBMPruningCallback(trial, "binary_logloss"),
                    ],
                )

                y_pred = model.predict(X_val)
                accuracy = ((y_pred > 0.5).astype(int) == y_val).mean()
                scores.append(accuracy)

            return np.mean(scores)

        # 最適化
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best_params = {**self.DEFAULT_PARAMS, **study.best_params}
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")

        self.params = best_params
        return best_params

    def evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        モデルを評価

        Args:
            X: 特徴量
            y: ターゲット

        Returns:
            評価指標辞書
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        y_pred_proba = self.model.predict(X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 精度
        accuracy = (y_pred == y).mean()

        # 適合率・再現率
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # ログロス
        from sklearn.metrics import log_loss
        logloss = log_loss(y, y_pred_proba)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log_loss": logloss,
        }

    def save_model(self, path: str) -> None:
        """モデルを保存"""
        if self.model is None:
            raise RuntimeError("No model to save")

        import pickle

        save_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "params": self.params,
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "training_history": self.training_history,
            },
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Model saved to {path}")


class RetrainingManager:
    """再学習管理"""

    def __init__(
        self,
        current_model_path: str,
        models_dir: str = "models",
        keep_previous: int = 3,
    ):
        """
        Args:
            current_model_path: 現在のモデルパス
            models_dir: モデル保存ディレクトリ
            keep_previous: 保持する過去モデル数
        """
        self.current_model_path = current_model_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.keep_previous = keep_previous

    def should_retrain(
        self,
        performance_metrics: Dict[str, float],
        thresholds: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        再学習が必要か判定

        Args:
            performance_metrics: 現在のパフォーマンス指標
            thresholds: 閾値設定

        Returns:
            (再学習必要フラグ, トリガー理由リスト)
        """
        triggers = []

        # 勝率チェック
        if "win_rate" in performance_metrics and "win_rate" in thresholds:
            if performance_metrics["win_rate"] < thresholds["win_rate"]:
                triggers.append(f"Win rate below threshold: {performance_metrics['win_rate']:.2%}")

        # シャープレシオチェック
        if "sharpe_ratio" in performance_metrics and "sharpe_ratio" in thresholds:
            if performance_metrics["sharpe_ratio"] < thresholds["sharpe_ratio"]:
                triggers.append(f"Sharpe ratio below threshold: {performance_metrics['sharpe_ratio']:.2f}")

        # 連敗チェック
        if "consecutive_losses" in performance_metrics and "consecutive_losses" in thresholds:
            if performance_metrics["consecutive_losses"] >= thresholds["consecutive_losses"]:
                triggers.append(f"Consecutive losses: {performance_metrics['consecutive_losses']}")

        return len(triggers) > 0, triggers

    def compare_models(
        self,
        current_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        improvement_threshold: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        モデルを比較

        Args:
            current_metrics: 現在のモデルの指標
            new_metrics: 新モデルの指標
            improvement_threshold: 改善閾値

        Returns:
            (切り替え推奨フラグ, 改善率)
        """
        # シャープレシオを主要指標として使用
        current_score = current_metrics.get("sharpe_ratio", 0)
        new_score = new_metrics.get("sharpe_ratio", 0)

        if current_score == 0:
            improvement = float("inf") if new_score > 0 else 0
        else:
            improvement = (new_score - current_score) / abs(current_score)

        should_switch = improvement >= improvement_threshold

        return should_switch, improvement

    def save_new_model(
        self,
        trainer: ModelTrainer,
        metrics: Dict[str, float],
    ) -> str:
        """
        新モデルを保存

        Args:
            trainer: 学習済みトレーナー
            metrics: 評価指標

        Returns:
            保存先パス
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = self.models_dir / f"model_{timestamp}.pkl"

        trainer.save_model(str(new_path))

        # 古いモデルを削除
        self._cleanup_old_models()

        return str(new_path)

    def _cleanup_old_models(self) -> None:
        """古いモデルを削除"""
        model_files = sorted(
            self.models_dir.glob("model_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        for old_model in model_files[self.keep_previous:]:
            old_model.unlink()
            logger.info(f"Removed old model: {old_model}")

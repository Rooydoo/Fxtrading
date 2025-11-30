"""
再学習トリガーモジュール
パフォーマンス劣化やドリフト検出時の再学習トリガー
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

import yaml

from .performance_tracker import PerformanceTracker
from .drift_detector import DriftDetector, PredictionDriftDetector

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """再学習トリガー"""

    def __init__(
        self,
        config_path: str = "config/retraining.yaml",
        performance_tracker: Optional[PerformanceTracker] = None,
        drift_detector: Optional[DriftDetector] = None,
        prediction_drift_detector: Optional[PredictionDriftDetector] = None,
    ):
        """
        Args:
            config_path: 設定ファイルパス
            performance_tracker: パフォーマンストラッカー
            drift_detector: 特徴量ドリフト検出
            prediction_drift_detector: 予測ドリフト検出
        """
        self.config = self._load_config(config_path)
        self.performance_tracker = performance_tracker
        self.drift_detector = drift_detector
        self.prediction_drift_detector = prediction_drift_detector

        self._last_check: Optional[datetime] = None
        self._triggered = False
        self._trigger_reasons: List[str] = []

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
            "triggers": {
                "win_rate": {"enabled": True, "period_days": 14, "threshold": 0.45},
                "sharpe_ratio": {"enabled": True, "period_days": 14, "threshold": 0.3},
                "prediction_accuracy": {"enabled": True, "threshold": 0.15},
                "feature_drift": {"enabled": True, "threshold": 0.25},
                "consecutive_loss": {"enabled": True, "threshold": 5},
            },
        }

    def check_triggers(self) -> Dict[str, Any]:
        """
        全トリガー条件をチェック

        Returns:
            チェック結果
        """
        self._trigger_reasons.clear()
        self._triggered = False

        results = {
            "timestamp": datetime.now().isoformat(),
            "triggered": False,
            "reasons": [],
            "checks": {},
        }

        triggers_config = self.config.get("triggers", {})

        # 勝率チェック
        if triggers_config.get("win_rate", {}).get("enabled", True):
            check_result = self._check_win_rate(triggers_config["win_rate"])
            results["checks"]["win_rate"] = check_result
            if check_result["triggered"]:
                self._trigger_reasons.append(check_result["reason"])

        # シャープレシオチェック
        if triggers_config.get("sharpe_ratio", {}).get("enabled", True):
            check_result = self._check_sharpe_ratio(triggers_config["sharpe_ratio"])
            results["checks"]["sharpe_ratio"] = check_result
            if check_result["triggered"]:
                self._trigger_reasons.append(check_result["reason"])

        # 予測精度チェック
        if triggers_config.get("prediction_accuracy", {}).get("enabled", True):
            check_result = self._check_prediction_accuracy(triggers_config["prediction_accuracy"])
            results["checks"]["prediction_accuracy"] = check_result
            if check_result["triggered"]:
                self._trigger_reasons.append(check_result["reason"])

        # 特徴量ドリフトチェック
        if triggers_config.get("feature_drift", {}).get("enabled", True):
            check_result = self._check_feature_drift(triggers_config["feature_drift"])
            results["checks"]["feature_drift"] = check_result
            if check_result["triggered"]:
                self._trigger_reasons.append(check_result["reason"])

        # 連敗チェック
        if triggers_config.get("consecutive_loss", {}).get("enabled", True):
            check_result = self._check_consecutive_loss(triggers_config["consecutive_loss"])
            results["checks"]["consecutive_loss"] = check_result
            if check_result["triggered"]:
                self._trigger_reasons.append(check_result["reason"])

        self._triggered = len(self._trigger_reasons) > 0
        results["triggered"] = self._triggered
        results["reasons"] = self._trigger_reasons

        self._last_check = datetime.now()

        if self._triggered:
            logger.warning(f"Retraining triggered: {self._trigger_reasons}")

        return results

    def _check_win_rate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """勝率チェック"""
        if not self.performance_tracker:
            return {"triggered": False, "error": "No performance tracker"}

        period_days = config.get("period_days", 14)
        threshold = config.get("threshold", 0.45)
        min_trades = config.get("min_trades", 10)

        start_date = datetime.now() - timedelta(days=period_days)
        metrics = self.performance_tracker.get_period_metrics(start_date)

        win_rate = metrics.get("win_rate", 0)
        total_trades = metrics.get("total_trades", 0)

        triggered = total_trades >= min_trades and win_rate < threshold

        return {
            "triggered": triggered,
            "current_value": win_rate,
            "threshold": threshold,
            "total_trades": total_trades,
            "reason": f"Win rate {win_rate:.1%} < {threshold:.1%}" if triggered else None,
        }

    def _check_sharpe_ratio(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """シャープレシオチェック"""
        if not self.performance_tracker:
            return {"triggered": False, "error": "No performance tracker"}

        threshold = config.get("threshold", 0.3)

        metrics = self.performance_tracker.get_metrics()
        sharpe = metrics.get("sharpe_ratio", 0)

        triggered = sharpe < threshold and metrics.get("total_trades", 0) >= 10

        return {
            "triggered": triggered,
            "current_value": sharpe,
            "threshold": threshold,
            "reason": f"Sharpe ratio {sharpe:.2f} < {threshold}" if triggered else None,
        }

    def _check_prediction_accuracy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """予測精度チェック"""
        if not self.prediction_drift_detector:
            return {"triggered": False, "error": "No prediction drift detector"}

        result = self.prediction_drift_detector.detect_drift()

        return {
            "triggered": result.get("drift_detected", False),
            "current_value": result.get("current_accuracy"),
            "baseline": result.get("baseline_accuracy"),
            "deviation": result.get("deviation"),
            "reason": f"Prediction accuracy deviation {result.get('deviation', 0):.1%}" if result.get("drift_detected") else None,
        }

    def _check_feature_drift(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """特徴量ドリフトチェック"""
        if not self.drift_detector:
            return {"triggered": False, "error": "No drift detector"}

        # 注: この関数を呼び出す前に、外部でdetect_driftを実行する必要がある
        # ここでは最後のドリフト検出結果を使用するか、
        # DataFrameを渡して検出を行う設計が必要

        return {
            "triggered": False,
            "reason": None,
            "note": "Feature drift check requires external data",
        }

    def _check_consecutive_loss(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """連敗チェック"""
        if not self.performance_tracker:
            return {"triggered": False, "error": "No performance tracker"}

        threshold = config.get("threshold", 5)

        metrics = self.performance_tracker.get_metrics()
        consecutive_losses = metrics.get("consecutive_losses", 0)

        triggered = consecutive_losses >= threshold

        return {
            "triggered": triggered,
            "current_value": consecutive_losses,
            "threshold": threshold,
            "reason": f"Consecutive losses {consecutive_losses} >= {threshold}" if triggered else None,
        }

    def should_check(self) -> bool:
        """
        チェックを実行すべきか判定

        Returns:
            チェック実行フラグ
        """
        schedule = self.config.get("schedule", {})
        frequency = schedule.get("check_frequency", "weekly")

        if self._last_check is None:
            return True

        now = datetime.now()

        if frequency == "daily":
            return (now - self._last_check) >= timedelta(days=1)
        elif frequency == "weekly":
            # 指定の曜日・時間にチェック
            check_day = schedule.get("check_day", "Sunday")
            check_time = schedule.get("check_time", "23:00")

            day_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
            }

            target_weekday = day_map.get(check_day, 6)
            target_hour = int(check_time.split(":")[0])

            if now.weekday() == target_weekday and now.hour >= target_hour:
                if self._last_check.date() != now.date():
                    return True

        return False

    @property
    def is_triggered(self) -> bool:
        """トリガー状態を取得"""
        return self._triggered

    @property
    def trigger_reasons(self) -> List[str]:
        """トリガー理由を取得"""
        return self._trigger_reasons

    def reset(self) -> None:
        """状態をリセット"""
        self._triggered = False
        self._trigger_reasons.clear()
        self._last_check = None


class RetrainingOrchestrator:
    """再学習オーケストレーター"""

    def __init__(
        self,
        trigger: RetrainingTrigger,
        retrain_callback: Callable[[], Dict[str, Any]],
        notify_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Args:
            trigger: 再学習トリガー
            retrain_callback: 再学習実行コールバック
            notify_callback: 通知コールバック
        """
        self.trigger = trigger
        self.retrain_callback = retrain_callback
        self.notify_callback = notify_callback

        self._last_retrain: Optional[datetime] = None
        self._retrain_history: List[Dict[str, Any]] = []

    def run_check_and_retrain(self) -> Dict[str, Any]:
        """
        チェックを実行し、必要に応じて再学習

        Returns:
            実行結果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "checked": False,
            "retrained": False,
        }

        # チェック実行判定
        if not self.trigger.should_check():
            result["skipped"] = True
            result["skip_reason"] = "Not scheduled"
            return result

        # トリガーチェック
        check_result = self.trigger.check_triggers()
        result["checked"] = True
        result["check_result"] = check_result

        if not check_result["triggered"]:
            result["retrained"] = False
            return result

        # 通知
        if self.notify_callback:
            self.notify_callback("trigger_detected", {
                "reasons": check_result["reasons"],
            })

        # 再学習実行
        try:
            retrain_result = self.retrain_callback()
            result["retrained"] = True
            result["retrain_result"] = retrain_result

            self._last_retrain = datetime.now()
            self._retrain_history.append({
                "timestamp": self._last_retrain.isoformat(),
                "reasons": check_result["reasons"],
                "result": retrain_result,
            })

            # 完了通知
            if self.notify_callback:
                self.notify_callback("retrain_completed", retrain_result)

        except Exception as e:
            logger.exception(f"Retraining failed: {e}")
            result["retrained"] = False
            result["error"] = str(e)

            if self.notify_callback:
                self.notify_callback("retrain_failed", {"error": str(e)})

        return result

    def get_retrain_history(self) -> List[Dict[str, Any]]:
        """再学習履歴を取得"""
        return self._retrain_history

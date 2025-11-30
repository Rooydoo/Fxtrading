"""
動作モード管理モジュール
LIVE/PAPER/BACKTEST モードの切り替え
"""
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """トレーディングモード"""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class ModeManager:
    """モード管理クラス"""

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self._mode: TradingMode = TradingMode.PAPER
        self._config: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self) -> None:
        """設定を読み込み"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)

            mode_str = self._config.get("system", {}).get("mode", "paper")
            self._mode = TradingMode(mode_str.lower())

            logger.info(f"Mode loaded from config: {self._mode.value}")

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            self._mode = TradingMode.PAPER

        except ValueError as e:
            logger.error(f"Invalid mode in config: {e}")
            self._mode = TradingMode.PAPER

    @property
    def mode(self) -> TradingMode:
        """現在のモードを取得"""
        return self._mode

    @mode.setter
    def mode(self, value: TradingMode) -> None:
        """モードを設定"""
        old_mode = self._mode
        self._mode = value
        logger.info(f"Mode changed: {old_mode.value} -> {value.value}")

    def is_live(self) -> bool:
        """ライブモードか"""
        return self._mode == TradingMode.LIVE

    def is_paper(self) -> bool:
        """ペーパーモードか"""
        return self._mode == TradingMode.PAPER

    def is_backtest(self) -> bool:
        """バックテストモードか"""
        return self._mode == TradingMode.BACKTEST

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得

        Args:
            key: 設定キー (ドット区切り)
            default: デフォルト値

        Returns:
            設定値
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def require_live_credentials(self) -> bool:
        """
        ライブ認証情報が必要か

        Returns:
            認証情報必要フラグ
        """
        return self._mode == TradingMode.LIVE


class EnvironmentManager:
    """環境変数管理"""

    REQUIRED_VARS = {
        TradingMode.LIVE: [
            "GMO_API_KEY",
            "GMO_API_SECRET",
        ],
        TradingMode.PAPER: [],
        TradingMode.BACKTEST: [],
    }

    OPTIONAL_VARS = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
    ]

    def __init__(self, mode: TradingMode):
        """
        Args:
            mode: トレーディングモード
        """
        self.mode = mode

    def validate(self) -> Dict[str, Any]:
        """
        環境変数を検証

        Returns:
            検証結果
        """
        required = self.REQUIRED_VARS.get(self.mode, [])
        missing = []
        found = []

        for var in required:
            if os.getenv(var):
                found.append(var)
            else:
                missing.append(var)

        optional_status = {}
        for var in self.OPTIONAL_VARS:
            optional_status[var] = bool(os.getenv(var))

        is_valid = len(missing) == 0

        result = {
            "valid": is_valid,
            "mode": self.mode.value,
            "required": {
                "found": found,
                "missing": missing,
            },
            "optional": optional_status,
        }

        if not is_valid:
            logger.error(f"Missing required environment variables: {missing}")

        return result

    def get_credentials(self) -> Dict[str, Optional[str]]:
        """
        認証情報を取得

        Returns:
            認証情報辞書
        """
        return {
            "api_key": os.getenv("GMO_API_KEY"),
            "api_secret": os.getenv("GMO_API_SECRET"),
            "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN"),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        }


class SystemState:
    """システム状態管理"""

    def __init__(self):
        self._state = "initialized"
        self._paused = False
        self._shutdown_requested = False
        self._error_state = False
        self._error_message: Optional[str] = None

    @property
    def state(self) -> str:
        """現在の状態"""
        if self._shutdown_requested:
            return "shutdown"
        if self._error_state:
            return "error"
        if self._paused:
            return "paused"
        return self._state

    def set_running(self) -> None:
        """実行中に設定"""
        self._state = "running"
        self._error_state = False
        logger.info("System state: running")

    def set_paused(self) -> None:
        """一時停止に設定"""
        self._paused = True
        logger.info("System state: paused")

    def resume(self) -> None:
        """再開"""
        self._paused = False
        logger.info("System state: resumed")

    def set_error(self, message: str) -> None:
        """エラー状態に設定"""
        self._error_state = True
        self._error_message = message
        logger.error(f"System state: error - {message}")

    def clear_error(self) -> None:
        """エラーをクリア"""
        self._error_state = False
        self._error_message = None
        logger.info("System error cleared")

    def request_shutdown(self) -> None:
        """シャットダウン要求"""
        self._shutdown_requested = True
        logger.info("Shutdown requested")

    def is_running(self) -> bool:
        """実行可能か"""
        return self.state == "running"

    def can_trade(self) -> bool:
        """取引可能か"""
        return self.state == "running" and not self._paused

    def get_status(self) -> Dict[str, Any]:
        """ステータスを取得"""
        return {
            "state": self.state,
            "paused": self._paused,
            "error": self._error_state,
            "error_message": self._error_message,
            "shutdown_requested": self._shutdown_requested,
        }

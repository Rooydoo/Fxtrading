"""
ポジション復旧モジュール
システム再起動時にオープンポジションを復元・同期
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionRecoveryManager:
    """ポジション復旧管理クラス"""

    def __init__(
        self,
        state_file: str = "data/position_state.json",
        backup_dir: str = "data/backups",
    ):
        """
        Args:
            state_file: 状態ファイルパス
            backup_dir: バックアップディレクトリ
        """
        self.state_file = Path(state_file)
        self.backup_dir = Path(backup_dir)

        # ディレクトリ作成
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def save_state(
        self,
        positions: List[Dict[str, Any]],
        system_state: Dict[str, Any],
        trailing_states: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        システム状態を保存

        Args:
            positions: オープンポジションリスト
            system_state: システム状態
            trailing_states: トレーリングストップ状態

        Returns:
            成功フラグ
        """
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "positions": positions,
                "system_state": system_state,
                "trailing_states": trailing_states or {},
                "version": "1.0",
            }

            # 既存ファイルをバックアップ
            if self.state_file.exists():
                self._create_backup()

            # 状態を保存
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            logger.info(f"State saved: {len(positions)} positions")
            return True

        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        システム状態を読み込み

        Returns:
            保存された状態、またはNone
        """
        if not self.state_file.exists():
            logger.info("No saved state found")
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            timestamp = state.get("timestamp")
            positions = state.get("positions", [])

            logger.info(f"State loaded: {len(positions)} positions (saved at {timestamp})")
            return state

        except json.JSONDecodeError as e:
            logger.error(f"Invalid state file: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def _create_backup(self) -> None:
        """バックアップを作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"position_state_{timestamp}.json"

            with open(self.state_file, "r") as src:
                with open(backup_file, "w") as dst:
                    dst.write(src.read())

            # 古いバックアップを削除 (10個まで保持)
            self._cleanup_backups(keep=10)

            logger.debug(f"Backup created: {backup_file}")

        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _cleanup_backups(self, keep: int = 10) -> None:
        """古いバックアップを削除"""
        backups = sorted(self.backup_dir.glob("position_state_*.json"))
        if len(backups) > keep:
            for old_backup in backups[:-keep]:
                try:
                    old_backup.unlink()
                except Exception:
                    pass

    def clear_state(self) -> bool:
        """状態ファイルを削除（正常終了時）"""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info("State file cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear state: {e}")
            return False

    def needs_recovery(self) -> bool:
        """復旧が必要か判定"""
        if not self.state_file.exists():
            return False

        state = self.load_state()
        if not state:
            return False

        positions = state.get("positions", [])
        return len(positions) > 0


class PositionSynchronizer:
    """ポジション同期クラス（API側との同期）"""

    def __init__(self, api_client):
        """
        Args:
            api_client: GMO APIクライアント
        """
        self.client = api_client

    def get_api_positions(self) -> List[Dict[str, Any]]:
        """
        API側のオープンポジションを取得

        Returns:
            ポジションリスト
        """
        try:
            response = self.client.get_open_positions()
            if response.get("status") == 0:
                return response.get("data", {}).get("list", [])
            else:
                logger.warning(f"Failed to get API positions: {response}")
                return []
        except Exception as e:
            logger.error(f"API error getting positions: {e}")
            return []

    def synchronize(
        self,
        local_positions: List[Dict[str, Any]],
        position_manager,
    ) -> Dict[str, Any]:
        """
        ローカルとAPI側のポジションを同期

        Args:
            local_positions: ローカル保存されたポジション
            position_manager: PositionManagerインスタンス

        Returns:
            同期結果
        """
        result = {
            "synced": [],
            "added": [],
            "removed": [],
            "conflicts": [],
        }

        # API側のポジションを取得
        api_positions = self.get_api_positions()
        api_position_ids = {p.get("positionId") for p in api_positions}
        local_position_ids = {p.get("position_id") for p in local_positions}

        # API側にあってローカルにないもの（追加）
        for api_pos in api_positions:
            pos_id = api_pos.get("positionId")
            if pos_id not in local_position_ids:
                logger.warning(f"Position found on API but not local: {pos_id}")
                result["added"].append(api_pos)

        # ローカルにあってAPI側にないもの（削除済み）
        for local_pos in local_positions:
            pos_id = local_pos.get("position_id")
            if pos_id not in api_position_ids:
                logger.warning(f"Position in local but not on API: {pos_id}")
                result["removed"].append(local_pos)

        # 両方にあるもの（同期）
        for local_pos in local_positions:
            pos_id = local_pos.get("position_id")
            if pos_id in api_position_ids:
                api_pos = next(
                    (p for p in api_positions if p.get("positionId") == pos_id),
                    None
                )
                if api_pos:
                    # サイズや価格の整合性チェック
                    if self._check_consistency(local_pos, api_pos):
                        result["synced"].append(local_pos)
                    else:
                        result["conflicts"].append({
                            "local": local_pos,
                            "api": api_pos,
                        })

        return result

    def _check_consistency(
        self,
        local_pos: Dict[str, Any],
        api_pos: Dict[str, Any],
    ) -> bool:
        """ポジションの整合性をチェック"""
        # サイズチェック
        local_size = local_pos.get("size", 0)
        api_size = int(api_pos.get("size", 0))

        if abs(local_size - api_size) > 1:
            logger.warning(
                f"Size mismatch: local={local_size}, api={api_size}"
            )
            return False

        # 方向チェック
        local_side = local_pos.get("side", "").upper()
        api_side = api_pos.get("side", "").upper()

        if local_side != api_side:
            logger.warning(
                f"Side mismatch: local={local_side}, api={api_side}"
            )
            return False

        return True

    def recover_positions(
        self,
        local_positions: List[Dict[str, Any]],
        position_manager,
    ) -> int:
        """
        ポジションを復旧してPositionManagerに登録

        Args:
            local_positions: 復旧するポジション
            position_manager: PositionManagerインスタンス

        Returns:
            復旧したポジション数
        """
        recovered = 0

        for pos_data in local_positions:
            try:
                # PositionManagerに登録
                position = position_manager.add_position_from_dict(pos_data)
                if position:
                    recovered += 1
                    logger.info(f"Position recovered: {position.position_id}")
            except Exception as e:
                logger.error(f"Failed to recover position: {e}")

        return recovered


class RecoveryHandler:
    """復旧ハンドラー（メイン復旧ロジック）"""

    def __init__(
        self,
        recovery_manager: PositionRecoveryManager,
        synchronizer: Optional[PositionSynchronizer] = None,
    ):
        """
        Args:
            recovery_manager: PositionRecoveryManager
            synchronizer: PositionSynchronizer (Live modeの場合)
        """
        self.recovery_manager = recovery_manager
        self.synchronizer = synchronizer

    def perform_recovery(
        self,
        position_manager,
        trailing_stop_manager=None,
        is_live: bool = False,
    ) -> Dict[str, Any]:
        """
        復旧を実行

        Args:
            position_manager: PositionManagerインスタンス
            trailing_stop_manager: TrailingStopManagerインスタンス
            is_live: 本番モードか

        Returns:
            復旧結果
        """
        result = {
            "success": False,
            "positions_recovered": 0,
            "trailing_states_recovered": 0,
            "warnings": [],
            "errors": [],
        }

        # 状態を読み込み
        state = self.recovery_manager.load_state()
        if not state:
            result["success"] = True  # 復旧不要
            return result

        local_positions = state.get("positions", [])
        trailing_states = state.get("trailing_states", {})

        # 本番モードの場合はAPI同期
        if is_live and self.synchronizer:
            sync_result = self.synchronizer.synchronize(
                local_positions, position_manager
            )

            # 競合チェック
            if sync_result["conflicts"]:
                result["warnings"].append(
                    f"Found {len(sync_result['conflicts'])} position conflicts"
                )

            # API側で決済済みのポジションを除外
            active_positions = [
                p for p in local_positions
                if p.get("position_id") not in
                   {r.get("position_id") for r in sync_result["removed"]}
            ]
        else:
            active_positions = local_positions

        # ポジションを復旧
        for pos_data in active_positions:
            try:
                position = position_manager.add_position_from_dict(pos_data)
                if position:
                    result["positions_recovered"] += 1
            except Exception as e:
                result["errors"].append(f"Position recovery failed: {e}")

        # トレーリングストップ状態を復旧
        if trailing_stop_manager and trailing_states:
            try:
                trailing_stop_manager.load_states(trailing_states)
                result["trailing_states_recovered"] = len(trailing_states)
            except Exception as e:
                result["errors"].append(f"Trailing state recovery failed: {e}")

        result["success"] = len(result["errors"]) == 0

        logger.info(
            f"Recovery completed: {result['positions_recovered']} positions, "
            f"{result['trailing_states_recovered']} trailing states"
        )

        return result

    def save_current_state(
        self,
        position_manager,
        system_state: Dict[str, Any],
        trailing_stop_manager=None,
    ) -> bool:
        """
        現在の状態を保存

        Args:
            position_manager: PositionManagerインスタンス
            system_state: システム状態
            trailing_stop_manager: TrailingStopManagerインスタンス

        Returns:
            成功フラグ
        """
        # ポジションをシリアライズ
        positions = [
            p.to_dict() for p in position_manager.get_open_positions()
        ]

        # トレーリング状態
        trailing_states = None
        if trailing_stop_manager:
            trailing_states = trailing_stop_manager.save_states()

        return self.recovery_manager.save_state(
            positions=positions,
            system_state=system_state,
            trailing_states=trailing_states,
        )

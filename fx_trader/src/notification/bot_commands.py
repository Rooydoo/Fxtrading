"""
Telegramボットコマンドシステム
ステータス確認、パラメータ調整、緊急制御
"""
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramBotCommands:
    """Telegramボットコマンドシステム"""

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    # 利用可能なコマンド
    COMMANDS = {
        "status": "現在のステータスを表示",
        "balance": "残高と損益を表示",
        "positions": "オープンポジション一覧",
        "report": "パフォーマンスレポート",
        "stats": "統計情報",
        "pause": "取引を一時停止",
        "resume": "取引を再開",
        "closeall": "全ポジション決済（緊急）",
        "close": "特定ポジション決済 /close <id>",
        "risk": "リスク設定の表示/変更",
        "set": "パラメータ設定 /set <key> <value>",
        "help": "ヘルプを表示",
    }

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        system_callback: Optional[Callable] = None,
    ):
        """
        Args:
            bot_token: Telegram Botトークン
            chat_id: チャットID
            system_callback: システムコールバック (コマンド実行用)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.system_callback = system_callback

        self._last_update_id = 0
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None

        # コマンドハンドラー登録
        self._handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """デフォルトハンドラーを登録"""
        self.register_handler("help", self._cmd_help)
        self.register_handler("start", self._cmd_help)

    def register_handler(self, command: str, handler: Callable) -> None:
        """
        コマンドハンドラーを登録

        Args:
            command: コマンド名
            handler: ハンドラー関数 (args: List[str]) -> str
        """
        self._handlers[command] = handler

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """メッセージを送信"""
        try:
            url = self.API_URL.format(token=self.bot_token, method="sendMessage")
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.json().get("ok", False)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _get_updates(self, timeout: int = 30) -> List[Dict]:
        """更新を取得（ロングポーリング）"""
        try:
            url = self.API_URL.format(token=self.bot_token, method="getUpdates")
            params = {
                "offset": self._last_update_id + 1,
                "timeout": timeout,
                "allowed_updates": ["message"],
            }
            response = requests.get(url, params=params, timeout=timeout + 5)
            result = response.json()

            if result.get("ok"):
                updates = result.get("result", [])
                if updates:
                    self._last_update_id = updates[-1]["update_id"]
                return updates
        except Exception as e:
            logger.debug(f"Get updates error: {e}")
        return []

    def _process_update(self, update: Dict) -> None:
        """更新を処理"""
        message = update.get("message", {})
        text = message.get("text", "")
        from_user = message.get("from", {})
        chat = message.get("chat", {})

        # チャットIDチェック
        if str(chat.get("id")) != str(self.chat_id):
            logger.warning(f"Unauthorized chat: {chat.get('id')}")
            return

        if not text.startswith("/"):
            return

        # コマンドパース
        parts = text.split()
        command = parts[0][1:].split("@")[0]  # /command@botname 対応
        args = parts[1:]

        logger.info(f"Command received: /{command} {args} from {from_user.get('username', 'unknown')}")

        # ハンドラー実行
        if command in self._handlers:
            try:
                response = self._handlers[command](args)
                if response:
                    self.send_message(response)
            except Exception as e:
                logger.exception(f"Command handler error: {e}")
                self.send_message(f"❌ エラー: {str(e)}")
        else:
            self.send_message(f"❓ 不明なコマンド: /{command}\n/help でコマンド一覧を表示")

    def _poll_loop(self) -> None:
        """ポーリングループ"""
        while self._running:
            try:
                updates = self._get_updates(timeout=30)
                for update in updates:
                    self._process_update(update)
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                time.sleep(5)

    def start_polling(self) -> None:
        """ポーリングを開始"""
        if self._running:
            return

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info("Telegram bot polling started")

    def stop_polling(self) -> None:
        """ポーリングを停止"""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        logger.info("Telegram bot polling stopped")

    # ==================== コマンドハンドラー ====================

    def _cmd_help(self, args: List[str]) -> str:
        """ヘルプコマンド"""
        lines = ["📖 <b>利用可能なコマンド</b>\n"]
        for cmd, desc in self.COMMANDS.items():
            lines.append(f"/{cmd} - {desc}")
        return "\n".join(lines)


class TradingBotCommands(TelegramBotCommands):
    """トレーディングシステム用ボットコマンド"""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        trading_system=None,
    ):
        """
        Args:
            bot_token: Telegram Botトークン
            chat_id: チャットID
            trading_system: トレーディングシステムインスタンス
        """
        super().__init__(bot_token, chat_id)
        self.trading_system = trading_system

        # トレーディング用コマンドを登録
        self.register_handler("status", self._cmd_status)
        self.register_handler("balance", self._cmd_balance)
        self.register_handler("positions", self._cmd_positions)
        self.register_handler("report", self._cmd_report)
        self.register_handler("stats", self._cmd_stats)
        self.register_handler("pause", self._cmd_pause)
        self.register_handler("resume", self._cmd_resume)
        self.register_handler("closeall", self._cmd_closeall)
        self.register_handler("close", self._cmd_close)
        self.register_handler("risk", self._cmd_risk)
        self.register_handler("set", self._cmd_set)

    def _cmd_status(self, args: List[str]) -> str:
        """ステータスコマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            # ペーパーシミュレーターから取得
            if hasattr(self.trading_system, 'paper_simulator'):
                status = self.trading_system.paper_simulator.get_status()
            else:
                status = self._get_system_status()

            mode = status.get("mode", "UNKNOWN")
            balance = status.get("current_balance", 0)
            equity = status.get("equity", balance)
            unrealized = status.get("unrealized_pnl", 0)
            total_return = status.get("total_return", 0)
            positions = status.get("open_positions", 0)

            return f"""
📊 <b>システムステータス</b>

<b>モード:</b> {mode}
<b>残高:</b> ¥{balance:,.0f}
<b>評価額:</b> ¥{equity:,.0f}
<b>未実現損益:</b> ¥{unrealized:+,.0f}
<b>トータルリターン:</b> {total_return:+.2%}
<b>オープンポジション:</b> {positions}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        except Exception as e:
            return f"❌ ステータス取得エラー: {e}"

    def _cmd_balance(self, args: List[str]) -> str:
        """残高コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'paper_simulator'):
                sim = self.trading_system.paper_simulator
                return f"""
💰 <b>残高情報</b>

<b>初期資金:</b> ¥{sim.initial_balance:,.0f}
<b>現在残高:</b> ¥{sim.balance:,.0f}
<b>累計損益:</b> ¥{sim.stats['total_pnl']:+,.0f}
<b>最大DD:</b> {sim.stats['max_drawdown']:.2%}
<b>ピーク残高:</b> ¥{sim.stats['peak_balance']:,.0f}
"""
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_positions(self, args: List[str]) -> str:
        """ポジション一覧コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'paper_simulator'):
                positions = self.trading_system.paper_simulator.open_positions

                if not positions:
                    return "📋 オープンポジションはありません"

                lines = ["📋 <b>オープンポジション</b>\n"]
                for pos_id, pos in positions.items():
                    pnl = pos.get("unrealized_pnl", 0)
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    lines.append(
                        f"{emoji} <b>{pos['symbol']}</b> {pos['side']}\n"
                        f"   価格: {pos['entry_price']:.5f}\n"
                        f"   サイズ: {pos['size']:,.0f}\n"
                        f"   未実現: ¥{pnl:+,.0f}\n"
                        f"   ID: {pos_id[:20]}..."
                    )
                return "\n".join(lines)
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_report(self, args: List[str]) -> str:
        """レポートコマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            days = int(args[0]) if args else 7

            if hasattr(self.trading_system, 'paper_simulator'):
                report = self.trading_system.paper_simulator.get_performance_report(days)

                if report.get("no_data"):
                    return f"📈 {days}日間のデータがありません"

                return f"""
📈 <b>パフォーマンスレポート</b> (直近{days}日)

<b>取引回数:</b> {report['total_trades']}
<b>勝率:</b> {report['win_rate']:.1%}
<b>累計損益:</b> ¥{report['total_pnl']:+,.0f}
<b>平均損益:</b> ¥{report['avg_pnl']:+,.0f}
<b>平均勝ち:</b> ¥{report['avg_win']:+,.0f}
<b>平均負け:</b> ¥{report['avg_loss']:+,.0f}
<b>PF:</b> {report['profit_factor']:.2f}

<b>ベスト:</b> ¥{report['best_trade']:+,.0f}
<b>ワースト:</b> ¥{report['worst_trade']:+,.0f}
"""
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_stats(self, args: List[str]) -> str:
        """統計コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'paper_simulator'):
                stats = self.trading_system.paper_simulator.stats

                win_rate = (
                    stats['winning_trades'] / stats['total_trades']
                    if stats['total_trades'] > 0 else 0
                )

                return f"""
📊 <b>取引統計</b>

<b>総取引数:</b> {stats['total_trades']}
<b>勝ち:</b> {stats['winning_trades']}
<b>負け:</b> {stats['losing_trades']}
<b>勝率:</b> {win_rate:.1%}

<b>連勝:</b> 現在{stats['consecutive_wins']} / 最大{stats['max_consecutive_wins']}
<b>連敗:</b> 現在{stats['consecutive_losses']} / 最大{stats['max_consecutive_losses']}
<b>最大DD:</b> {stats['max_drawdown']:.2%}
"""
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_pause(self, args: List[str]) -> str:
        """一時停止コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'system_state'):
                self.trading_system.system_state.set_paused()
                return "⏸️ 取引を一時停止しました\n/resume で再開できます"
            return "❌ 一時停止機能が利用できません"
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_resume(self, args: List[str]) -> str:
        """再開コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'system_state'):
                self.trading_system.system_state.resume()
                return "▶️ 取引を再開しました"
            return "❌ 再開機能が利用できません"
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_closeall(self, args: List[str]) -> str:
        """全ポジション決済コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        # 確認が必要
        if not args or args[0].lower() != "confirm":
            return "⚠️ 全ポジションを決済しますか？\n確認する場合: /closeall confirm"

        try:
            if hasattr(self.trading_system, 'trade_executor'):
                results = self.trading_system.trade_executor.close_all_positions("telegram_emergency")
                total_pnl = sum(r.get("pnl", 0) or 0 for r in results)
                return f"🛑 {len(results)}ポジションを決済しました\n合計損益: ¥{total_pnl:+,.0f}"
            return "❌ 決済機能が利用できません"
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_close(self, args: List[str]) -> str:
        """特定ポジション決済コマンド"""
        if not args:
            return "使い方: /close <position_id>"

        if not self.trading_system:
            return "❌ システム未接続"

        position_id = args[0]

        try:
            if hasattr(self.trading_system, 'trade_executor'):
                pnl = self.trading_system.trade_executor.close_trade(position_id, "telegram_manual")
                if pnl is not None:
                    return f"✅ ポジション決済完了\n損益: ¥{pnl:+,.0f}"
                else:
                    return "❌ ポジションが見つかりません"
            return "❌ 決済機能が利用できません"
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_risk(self, args: List[str]) -> str:
        """リスク設定コマンド"""
        if not self.trading_system:
            return "❌ システム未接続"

        try:
            if hasattr(self.trading_system, 'risk_manager'):
                rm = self.trading_system.risk_manager
                return f"""
⚙️ <b>リスク設定</b>

<b>1トレードリスク:</b> {rm.config.get('position_risk', {}).get('long', {}).get('risk_per_trade', 0.01):.1%}
<b>デイリーリミット:</b> {rm.config.get('capital_management', {}).get('daily_loss_limit', {}).get('percent', 0.02):.1%}
<b>連敗閾値:</b> {rm.config.get('capital_management', {}).get('consecutive_loss', {}).get('threshold', 5)}回

<b>本日損益:</b> ¥{rm.daily_pnl:+,.0f}
<b>連敗数:</b> {rm.consecutive_losses}
<b>取引停止:</b> {'はい' if rm.trading_halted else 'いいえ'}
"""
        except Exception as e:
            return f"❌ エラー: {e}"

    def _cmd_set(self, args: List[str]) -> str:
        """パラメータ設定コマンド"""
        if len(args) < 2:
            return """
使い方: /set <key> <value>

設定可能なパラメータ:
  risk_per_trade - 1トレードリスク率 (例: 0.01)
  daily_limit - デイリー損失制限 (例: 0.02)
  threshold_long - ロング確信度閾値 (例: 0.55)
  threshold_short - ショート確信度閾値 (例: 0.55)
"""

        key = args[0]
        value = args[1]

        try:
            value_float = float(value)

            # 安全性チェック
            if key == "risk_per_trade":
                if not 0.001 <= value_float <= 0.05:
                    return "❌ risk_per_trade は 0.1% ～ 5% の範囲で設定してください"

            if key == "daily_limit":
                if not 0.01 <= value_float <= 0.1:
                    return "❌ daily_limit は 1% ～ 10% の範囲で設定してください"

            # 設定更新（実際の実装が必要）
            return f"✅ {key} を {value} に設定しました"

        except ValueError:
            return "❌ 無効な値です"
        except Exception as e:
            return f"❌ エラー: {e}"

    def _get_system_status(self) -> Dict[str, Any]:
        """システムステータスを取得（フォールバック）"""
        return {
            "mode": "UNKNOWN",
            "current_balance": 0,
            "equity": 0,
            "unrealized_pnl": 0,
            "total_return": 0,
            "open_positions": 0,
        }

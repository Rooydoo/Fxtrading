"""
Telegram通知モジュール
取引通知、アラート、定期レポートの送信
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram通知クラス"""

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Args:
            bot_token: Telegram Botトークン (環境変数 TELEGRAM_BOT_TOKEN)
            chat_id: 送信先チャットID (環境変数 TELEGRAM_CHAT_ID)
            enabled: 通知有効フラグ
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = enabled

        if self.enabled and (not self.bot_token or not self.chat_id):
            logger.warning("Telegram credentials not configured, notifications disabled")
            self.enabled = False

    def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        メッセージを送信

        Args:
            text: メッセージ本文
            parse_mode: パースモード (HTML, Markdown)
            disable_notification: 通知音を無効化

        Returns:
            送信成功フラグ
        """
        if not self.enabled:
            logger.debug(f"[Telegram disabled] {text}")
            return False

        try:
            url = self.API_URL.format(token=self.bot_token, method="sendMessage")

            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification,
            }

            response = requests.post(url, json=payload, timeout=10)
            result = response.json()

            if result.get("ok"):
                logger.debug("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {result}")
                return False

        except Exception as e:
            logger.exception(f"Failed to send Telegram message: {e}")
            return False

    def send_entry_notification(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        max_loss_amount: float,
        max_loss_percent: float,
    ) -> bool:
        """
        エントリー通知を送信

        Args:
            symbol: 通貨ペア
            side: 売買方向
            entry_price: エントリー価格
            size: ポジションサイズ
            stop_loss: ストップロス
            take_profit: テイクプロフィット
            confidence: 確信度
            max_loss_amount: 最大損失額
            max_loss_percent: 最大損失率
        """
        direction_emoji = "🔼" if side == "BUY" else "🔽"
        side_text = "LONG" if side == "BUY" else "SHORT"

        message = f"""
{direction_emoji} <b>新規エントリー</b>

<b>通貨ペア:</b> {symbol}
<b>方向:</b> {side_text}
<b>エントリー価格:</b> {entry_price:.5f}
<b>サイズ:</b> {size:,.0f}

<b>SL:</b> {stop_loss:.5f}
<b>TP:</b> {take_profit:.5f}

<b>確信度:</b> {confidence:.1%}
<b>最大損失:</b> ¥{max_loss_amount:,.0f} ({max_loss_percent:.1%})

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_exit_notification(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        pnl_pips: float,
        hold_time: str,
        reason: str,
    ) -> bool:
        """
        決済通知を送信

        Args:
            symbol: 通貨ペア
            side: 売買方向
            entry_price: エントリー価格
            exit_price: 決済価格
            size: ポジションサイズ
            pnl: 損益額
            pnl_pips: 損益(pips)
            hold_time: 保有時間
            reason: 決済理由
        """
        if pnl >= 0:
            result_emoji = "✅"
            result_text = "利益確定"
        else:
            result_emoji = "❌"
            result_text = "損切り"

        reason_text = {
            "take_profit": "TP到達",
            "stop_loss": "SL到達",
            "manual": "手動決済",
            "emergency": "緊急決済",
        }.get(reason, reason)

        message = f"""
{result_emoji} <b>ポジション決済</b>

<b>通貨ペア:</b> {symbol}
<b>方向:</b> {"LONG" if side == "BUY" else "SHORT"}
<b>決済理由:</b> {reason_text}

<b>エントリー:</b> {entry_price:.5f}
<b>決済:</b> {exit_price:.5f}
<b>サイズ:</b> {size:,.0f}

<b>損益:</b> ¥{pnl:+,.0f} ({pnl_pips:+.1f} pips)
<b>保有時間:</b> {hold_time}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())

    def send_emergency_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        緊急アラートを送信

        Args:
            alert_type: アラートタイプ
            message: アラートメッセージ
            details: 詳細情報
        """
        alert_text = f"""
🚨 <b>緊急アラート</b> 🚨

<b>タイプ:</b> {alert_type}
<b>メッセージ:</b> {message}
"""
        if details:
            for key, value in details.items():
                alert_text += f"<b>{key}:</b> {value}\n"

        alert_text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(alert_text.strip(), disable_notification=False)

    def send_daily_report(
        self,
        date: str,
        trades_count: int,
        wins: int,
        losses: int,
        total_pnl: float,
        cumulative_pnl: float,
        positions: List[Dict],
    ) -> bool:
        """
        日次レポートを送信

        Args:
            date: 日付
            trades_count: 取引数
            wins: 勝ち数
            losses: 負け数
            total_pnl: 当日損益
            cumulative_pnl: 累計損益
            positions: オープンポジション
        """
        win_rate = wins / trades_count if trades_count > 0 else 0

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        positions_text = ""
        if positions:
            positions_text = "\n<b>オープンポジション:</b>\n"
            for pos in positions:
                positions_text += f"  • {pos['symbol']} {pos['side']} @ {pos['entry_price']:.5f}\n"
        else:
            positions_text = "\n<b>オープンポジション:</b> なし"

        message = f"""
📊 <b>日次レポート</b> - {date}

<b>取引回数:</b> {trades_count}
<b>勝敗:</b> {wins}勝 {losses}敗 ({win_rate:.1%})

{pnl_emoji} <b>本日損益:</b> ¥{total_pnl:+,.0f}
<b>累計損益:</b> ¥{cumulative_pnl:+,.0f}
{positions_text}
"""
        return self.send_message(message.strip())

    def send_weekly_report(
        self,
        week: str,
        trades_count: int,
        win_rate: float,
        total_pnl: float,
        max_drawdown: float,
        sharpe_ratio: float,
        best_trade: float,
        worst_trade: float,
        hourly_performance: Optional[Dict[int, float]] = None,
    ) -> bool:
        """
        週次レポートを送信

        Args:
            week: 週番号/期間
            trades_count: 取引数
            win_rate: 勝率
            total_pnl: 週間損益
            max_drawdown: 最大ドローダウン
            sharpe_ratio: シャープレシオ
            best_trade: 最大利益
            worst_trade: 最大損失
            hourly_performance: 時間帯別パフォーマンス
        """
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"

        message = f"""
📋 <b>週次レポート</b> - {week}

<b>取引回数:</b> {trades_count}
<b>勝率:</b> {win_rate:.1%}

{pnl_emoji} <b>週間損益:</b> ¥{total_pnl:+,.0f}
<b>最大DD:</b> {max_drawdown:.1%}
<b>シャープレシオ:</b> {sharpe_ratio:.2f}

<b>ベストトレード:</b> ¥{best_trade:+,.0f}
<b>ワーストトレード:</b> ¥{worst_trade:+,.0f}
"""

        if hourly_performance:
            best_hour = max(hourly_performance, key=hourly_performance.get)
            worst_hour = min(hourly_performance, key=hourly_performance.get)
            message += f"""
<b>最も良い時間帯:</b> {best_hour}時
<b>最も悪い時間帯:</b> {worst_hour}時
"""

        return self.send_message(message.strip())

    def send_monthly_report(
        self,
        month: str,
        initial_balance: float,
        final_balance: float,
        total_return: float,
        trades_count: int,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
        sharpe_ratio: float,
    ) -> bool:
        """
        月次レポートを送信

        Args:
            month: 月
            initial_balance: 月初残高
            final_balance: 月末残高
            total_return: トータルリターン
            trades_count: 取引数
            win_rate: 勝率
            profit_factor: プロフィットファクター
            max_drawdown: 最大ドローダウン
            sharpe_ratio: シャープレシオ
        """
        return_emoji = "📈" if total_return >= 0 else "📉"

        message = f"""
📅 <b>月次レポート</b> - {month}

<b>月初残高:</b> ¥{initial_balance:,.0f}
<b>月末残高:</b> ¥{final_balance:,.0f}

{return_emoji} <b>月間リターン:</b> {total_return:+.1%}

<b>取引回数:</b> {trades_count}
<b>勝率:</b> {win_rate:.1%}
<b>PF:</b> {profit_factor:.2f}
<b>最大DD:</b> {max_drawdown:.1%}
<b>シャープレシオ:</b> {sharpe_ratio:.2f}
"""
        return self.send_message(message.strip())

    def send_model_update_notification(
        self,
        update_type: str,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        improvement: float,
    ) -> bool:
        """
        モデル更新通知を送信

        Args:
            update_type: 更新タイプ (retrained, switched, rollback)
            old_metrics: 旧モデル指標
            new_metrics: 新モデル指標
            improvement: 改善率
        """
        emoji_map = {
            "retrained": "🔄",
            "switched": "✅",
            "rollback": "⚠️",
            "no_improvement": "ℹ️",
        }
        emoji = emoji_map.get(update_type, "📌")

        message = f"""
{emoji} <b>モデル更新</b>

<b>タイプ:</b> {update_type}
<b>改善率:</b> {improvement:+.1%}

<b>旧モデル:</b>
  • Sharpe: {old_metrics.get('sharpe_ratio', 0):.2f}
  • 勝率: {old_metrics.get('win_rate', 0):.1%}

<b>新モデル:</b>
  • Sharpe: {new_metrics.get('sharpe_ratio', 0):.2f}
  • 勝率: {new_metrics.get('win_rate', 0):.1%}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message.strip())


class TelegramCommandHandler:
    """Telegramコマンドハンドラー"""

    def __init__(self, notifier: TelegramNotifier):
        """
        Args:
            notifier: TelegramNotifierインスタンス
        """
        self.notifier = notifier
        self._last_update_id = 0

    def get_updates(self) -> List[Dict[str, Any]]:
        """
        更新を取得

        Returns:
            メッセージリスト
        """
        if not self.notifier.enabled:
            return []

        try:
            url = TelegramNotifier.API_URL.format(
                token=self.notifier.bot_token,
                method="getUpdates",
            )

            params = {
                "offset": self._last_update_id + 1,
                "timeout": 1,
            }

            response = requests.get(url, params=params, timeout=5)
            result = response.json()

            if result.get("ok"):
                updates = result.get("result", [])
                if updates:
                    self._last_update_id = updates[-1]["update_id"]
                return updates

        except Exception as e:
            logger.warning(f"Failed to get updates: {e}")

        return []

    def parse_command(self, message: str) -> tuple:
        """
        コマンドをパース

        Args:
            message: メッセージテキスト

        Returns:
            (コマンド, 引数リスト)
        """
        if not message.startswith("/"):
            return None, []

        parts = message.split()
        command = parts[0][1:]  # /を除去
        args = parts[1:]

        return command, args

    def handle_commands(self, callback) -> None:
        """
        コマンドを処理

        Args:
            callback: コマンド処理コールバック関数
        """
        updates = self.get_updates()

        for update in updates:
            message = update.get("message", {})
            text = message.get("text", "")

            command, args = self.parse_command(text)
            if command:
                try:
                    callback(command, args)
                except Exception as e:
                    logger.error(f"Command handler error: {e}")

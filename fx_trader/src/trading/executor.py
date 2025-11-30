"""
売買実行モジュール
注文の発注、決済、ペーパートレード対応
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..data.gmo_client import GMOForexClient
from .position import Position, PositionManager, Side, TradeHistory
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class OrderExecutorBase(ABC):
    """注文執行の基底クラス"""

    @abstractmethod
    def execute_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """注文を執行"""
        pass

    @abstractmethod
    def close_position(
        self,
        position: Position,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """ポジションを決済"""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """現在価格を取得 (bid, ask)"""
        pass


class LiveOrderExecutor(OrderExecutorBase):
    """本番用注文執行"""

    def __init__(self, client: GMOForexClient):
        """
        Args:
            client: GMO APIクライアント
        """
        self.client = client

    def execute_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        注文を執行

        Args:
            symbol: 通貨ペア
            side: 売買方向
            size: 数量
            order_type: 注文タイプ (MARKET, LIMIT, STOP)
            price: 指値価格

        Returns:
            注文結果
        """
        try:
            response = self.client.create_order(
                symbol=symbol,
                side=side.value,
                size=str(int(size)),
                execution_type=order_type,
                price=str(price) if price else None,
            )

            if response.get("status") == 0:
                logger.info(f"Order executed: {symbol} {side.value} {size}")
                return {
                    "success": True,
                    "order_id": response.get("data"),
                    "response": response,
                }
            else:
                logger.error(f"Order failed: {response}")
                return {
                    "success": False,
                    "error": response.get("messages", []),
                    "response": response,
                }

        except Exception as e:
            logger.exception(f"Order execution error: {e}")
            return {"success": False, "error": str(e)}

    def close_position(
        self,
        position: Position,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        ポジションを決済

        Args:
            position: 決済するポジション
            price: 指値価格 (Noneで成行)

        Returns:
            決済結果
        """
        # 反対方向
        close_side = Side.SHORT if position.side == Side.LONG else Side.LONG

        try:
            response = self.client.close_order(
                symbol=position.symbol,
                side=close_side.value,
                size=str(int(position.size)),
                position_id=position.id,
                execution_type="MARKET" if price is None else "LIMIT",
                price=str(price) if price else None,
            )

            if response.get("status") == 0:
                logger.info(f"Position closed: {position.id}")
                return {"success": True, "response": response}
            else:
                logger.error(f"Close failed: {response}")
                return {"success": False, "error": response.get("messages", [])}

        except Exception as e:
            logger.exception(f"Close execution error: {e}")
            return {"success": False, "error": str(e)}

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """
        現在価格を取得

        Args:
            symbol: 通貨ペア

        Returns:
            (bid, ask)
        """
        response = self.client.get_ticker(symbol)

        if response.get("status") != 0 or not response.get("data"):
            raise Exception(f"Failed to get ticker: {response}")

        data = response["data"][0]
        return float(data["bid"]), float(data["ask"])


class PaperOrderExecutor(OrderExecutorBase):
    """ペーパートレード用注文執行"""

    def __init__(
        self,
        client: GMOForexClient,
        initial_balance: float = 1000000,
        slippage_pips: float = 0.5,
    ):
        """
        Args:
            client: GMO APIクライアント (価格取得用)
            initial_balance: 初期残高
            slippage_pips: スリッページ (pips)
        """
        self.client = client
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.slippage_pips = slippage_pips
        self._order_counter = 0

    def execute_order(
        self,
        symbol: str,
        side: Side,
        size: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        ペーパートレード注文を執行

        Args:
            symbol: 通貨ペア
            side: 売買方向
            size: 数量
            order_type: 注文タイプ
            price: 指値価格

        Returns:
            注文結果
        """
        try:
            bid, ask = self.get_current_price(symbol)

            # 約定価格決定
            if order_type == "MARKET":
                if side == Side.LONG:
                    fill_price = ask + self._calculate_slippage(symbol)
                else:
                    fill_price = bid - self._calculate_slippage(symbol)
            else:
                fill_price = price or (ask if side == Side.LONG else bid)

            self._order_counter += 1
            order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter}"

            logger.info(f"[PAPER] Order executed: {symbol} {side.value} {size} @ {fill_price}")

            return {
                "success": True,
                "order_id": order_id,
                "fill_price": fill_price,
                "size": size,
                "paper_trade": True,
            }

        except Exception as e:
            logger.exception(f"[PAPER] Order error: {e}")
            return {"success": False, "error": str(e)}

    def close_position(
        self,
        position: Position,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        ペーパートレードポジションを決済

        Args:
            position: 決済するポジション
            price: 指値価格

        Returns:
            決済結果
        """
        try:
            bid, ask = self.get_current_price(position.symbol)

            # 決済価格決定
            if price is None:
                if position.side == Side.LONG:
                    exit_price = bid - self._calculate_slippage(position.symbol)
                else:
                    exit_price = ask + self._calculate_slippage(position.symbol)
            else:
                exit_price = price

            # 損益計算
            pnl = position.calculate_pnl(exit_price)
            self.balance += pnl

            logger.info(f"[PAPER] Position closed: {position.id} @ {exit_price}, PnL={pnl:.2f}")

            return {
                "success": True,
                "exit_price": exit_price,
                "pnl": pnl,
                "paper_trade": True,
            }

        except Exception as e:
            logger.exception(f"[PAPER] Close error: {e}")
            return {"success": False, "error": str(e)}

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        """現在価格を取得"""
        response = self.client.get_ticker(symbol)

        if response.get("status") != 0 or not response.get("data"):
            raise Exception(f"Failed to get ticker: {response}")

        data = response["data"][0]
        return float(data["bid"]), float(data["ask"])

    def _calculate_slippage(self, symbol: str) -> float:
        """スリッページを計算"""
        if "JPY" in symbol:
            return self.slippage_pips / 100  # 0.005
        else:
            return self.slippage_pips / 10000  # 0.00005

    def get_balance(self) -> float:
        """現在残高を取得"""
        return self.balance

    def get_total_return(self) -> float:
        """トータルリターンを計算"""
        return (self.balance - self.initial_balance) / self.initial_balance

    def reset(self) -> None:
        """残高をリセット"""
        self.balance = self.initial_balance
        self._order_counter = 0
        logger.info(f"[PAPER] Balance reset to {self.initial_balance}")


class TradeExecutor:
    """統合取引執行クラス"""

    def __init__(
        self,
        executor: OrderExecutorBase,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        trade_history: TradeHistory,
    ):
        """
        Args:
            executor: 注文執行オブジェクト
            position_manager: ポジション管理
            risk_manager: リスク管理
            trade_history: 取引履歴
        """
        self.executor = executor
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.trade_history = trade_history

    def open_trade(
        self,
        symbol: str,
        side: Side,
        confidence: float,
        atr: float,
        balance: float,
    ) -> Optional[Position]:
        """
        新規取引を開始

        Args:
            symbol: 通貨ペア
            side: 売買方向
            confidence: モデル確信度
            atr: ATR値
            balance: 口座残高

        Returns:
            開設されたポジション (失敗時はNone)
        """
        # 取引可能かチェック
        can_trade, reason = self.risk_manager.can_trade(balance)
        if not can_trade:
            logger.warning(f"Cannot trade: {reason}")
            return None

        # ポジション上限チェック
        if not self.position_manager.can_open_position(symbol):
            logger.warning(f"Position limit reached for {symbol}")
            return None

        # 現在価格取得
        try:
            bid, ask = self.executor.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return None

        entry_price = ask if side == Side.LONG else bid

        # SL/TP計算
        stop_loss = self.risk_manager.calculate_stop_loss(side, entry_price, atr, symbol)
        take_profit = self.risk_manager.calculate_take_profit(side, entry_price, atr, symbol)

        # ポジションサイズ計算
        position_size = self.risk_manager.calculate_position_size(
            balance, side, entry_price, stop_loss, symbol
        )

        # 最大損失計算
        max_loss_amount, max_loss_percent = self.risk_manager.calculate_max_loss(
            balance, position_size, entry_price, stop_loss, symbol
        )

        # 注文執行
        result = self.executor.execute_order(symbol, side, position_size)

        if not result.get("success"):
            logger.error(f"Order failed: {result.get('error')}")
            return None

        # 約定価格を取得 (ペーパートレードの場合)
        fill_price = result.get("fill_price", entry_price)

        # ポジション登録
        position = self.position_manager.open_position(
            symbol=symbol,
            side=side,
            size=position_size,
            entry_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            max_loss_amount=max_loss_amount,
            metadata={
                "atr": atr,
                "order_id": result.get("order_id"),
                "paper_trade": result.get("paper_trade", False),
            },
        )

        return position

    def close_trade(
        self,
        position_id: str,
        reason: str = "manual",
    ) -> Optional[float]:
        """
        取引を決済

        Args:
            position_id: ポジションID
            reason: 決済理由

        Returns:
            損益 (失敗時はNone)
        """
        position = self.position_manager.get_position(position_id)
        if position is None:
            logger.warning(f"Position not found: {position_id}")
            return None

        # 決済執行
        result = self.executor.close_position(position)

        if not result.get("success"):
            logger.error(f"Close failed: {result.get('error')}")
            return None

        exit_price = result.get("exit_price")
        if exit_price is None:
            # Live取引の場合は現在価格を使用
            bid, ask = self.executor.get_current_price(position.symbol)
            exit_price = bid if position.side == Side.LONG else ask

        # ポジションを閉じる
        pnl = self.position_manager.close_position(position_id, exit_price, reason)

        # 取引履歴に保存
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason
        self.trade_history.save_trade(position)

        # リスク管理を更新
        self.risk_manager.update_trade_result(pnl)

        return pnl

    def check_and_close_positions(
        self,
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        全ポジションのSL/TPをチェックし、必要に応じて決済

        Args:
            prices: 通貨ペアごとの現在価格

        Returns:
            決済されたポジション情報リスト
        """
        to_close = self.position_manager.check_sl_tp(prices)
        results = []

        for close_info in to_close:
            position_id = close_info["position_id"]
            reason = close_info["reason"]

            pnl = self.close_trade(position_id, reason)

            if pnl is not None:
                results.append({
                    "position_id": position_id,
                    "reason": reason,
                    "pnl": pnl,
                })

        return results

    def close_all_positions(self, reason: str = "emergency") -> List[Dict[str, Any]]:
        """
        全ポジションを決済

        Args:
            reason: 決済理由

        Returns:
            決済結果リスト
        """
        results = []
        positions = self.position_manager.get_open_positions()

        for position in positions:
            pnl = self.close_trade(position.id, reason)
            results.append({
                "position_id": position.id,
                "symbol": position.symbol,
                "pnl": pnl,
            })

        return results

    def get_account_summary(self, balance: float) -> Dict[str, Any]:
        """
        口座サマリーを取得

        Args:
            balance: 口座残高

        Returns:
            サマリー情報
        """
        positions = self.position_manager.get_open_positions()

        # 現在価格を取得
        prices = {}
        for pos in positions:
            try:
                bid, ask = self.executor.get_current_price(pos.symbol)
                prices[pos.symbol] = (bid + ask) / 2
            except Exception:
                prices[pos.symbol] = pos.entry_price

        # リスクサマリー
        risk_summary = self.risk_manager.get_risk_summary(balance, positions, prices)

        # 取引統計
        stats = self.trade_history.get_statistics()

        return {
            "balance": balance,
            "positions": [p.to_dict() for p in positions],
            "risk": risk_summary,
            "statistics": stats,
        }

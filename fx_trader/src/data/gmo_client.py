"""
GMOコイン外国為替FX APIクライアント
https://forex-api.coin.z.com/
"""
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import os
import logging

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class GMOForexClient:
    """GMOコイン外国為替FX APIクライアント"""

    PUBLIC_ENDPOINT = "https://forex-api.coin.z.com/public"
    PRIVATE_ENDPOINT = "https://forex-api.coin.z.com/private"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Args:
            api_key: APIキー (環境変数 GMO_API_KEY からも取得可能)
            api_secret: APIシークレット (環境変数 GMO_API_SECRET からも取得可能)
            timeout: リクエストタイムアウト秒数
        """
        self.api_key = api_key or os.getenv("GMO_API_KEY")
        self.api_secret = api_secret or os.getenv("GMO_API_SECRET")
        self.timeout = timeout

        # レート制限管理
        self._last_get_time = 0.0
        self._last_post_time = 0.0
        self._get_interval = 1.0 / 6  # 1秒間に6回
        self._post_interval = 1.0  # 1秒間に1回

        # セッション設定
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """リトライ機能付きセッションを作成"""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _wait_rate_limit(self, is_post: bool = False) -> None:
        """レート制限を遵守するための待機"""
        current_time = time.time()
        if is_post:
            elapsed = current_time - self._last_post_time
            if elapsed < self._post_interval:
                time.sleep(self._post_interval - elapsed)
            self._last_post_time = time.time()
        else:
            elapsed = current_time - self._last_get_time
            if elapsed < self._get_interval:
                time.sleep(self._get_interval - elapsed)
            self._last_get_time = time.time()

    def _create_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """HMAC-SHA256署名を生成"""
        if not self.api_secret:
            raise ValueError("API secret is required for private endpoints")

        text = timestamp + method + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """認証ヘッダーを生成"""
        timestamp = str(int(time.time() * 1000))
        signature = self._create_signature(timestamp, method, path, body)

        return {
            "API-KEY": self.api_key or "",
            "API-TIMESTAMP": timestamp,
            "API-SIGN": signature,
            "Content-Type": "application/json",
        }

    # ==================== Public API ====================

    def get_status(self) -> Dict[str, Any]:
        """取引所ステータス取得"""
        self._wait_rate_limit()
        response = self.session.get(
            f"{self.PUBLIC_ENDPOINT}/v1/status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        最新レート取得

        Args:
            symbol: 通貨ペア (USD_JPY, EUR_USD など)
        """
        self._wait_rate_limit()
        response = self.session.get(
            f"{self.PUBLIC_ENDPOINT}/v1/ticker",
            params={"symbol": symbol},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ローソク足データ取得

        Args:
            symbol: 通貨ペア
            interval: 時間足 (1min, 5min, 10min, 15min, 30min, 1hour, 4hour, 8hour, 12hour, 1day, 1week, 1month)
            date: 日付 (YYYYMMDD形式、省略時は当日)
        """
        self._wait_rate_limit()
        params = {"symbol": symbol, "interval": interval}
        if date:
            params["date"] = date

        response = self.session.get(
            f"{self.PUBLIC_ENDPOINT}/v1/klines",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ==================== Private API ====================

    def get_account_margin(self) -> Dict[str, Any]:
        """口座余力情報取得"""
        self._wait_rate_limit()
        path = "/v1/account/margin"
        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_account_assets(self) -> Dict[str, Any]:
        """資産残高取得"""
        self._wait_rate_limit()
        path = "/v1/account/assets"
        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        建玉一覧取得

        Args:
            symbol: 通貨ペア (省略時は全通貨ペア)
        """
        self._wait_rate_limit()
        path = "/v1/openPositions"
        params = {}
        if symbol:
            params["symbol"] = symbol

        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_active_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        有効注文一覧取得

        Args:
            symbol: 通貨ペア (省略時は全通貨ペア)
        """
        self._wait_rate_limit()
        path = "/v1/activeOrders"
        params = {}
        if symbol:
            params["symbol"] = symbol

        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def create_order(
        self,
        symbol: str,
        side: str,
        size: str,
        execution_type: str = "MARKET",
        price: Optional[str] = None,
        loss_cut_price: Optional[str] = None,
        time_in_force: str = "FAK",
    ) -> Dict[str, Any]:
        """
        新規注文

        Args:
            symbol: 通貨ペア
            side: BUY or SELL
            size: 注文数量
            execution_type: MARKET, LIMIT, STOP
            price: 指値価格 (LIMIT/STOP時必須)
            loss_cut_price: ロスカットレート
            time_in_force: FAK, FAS, FOK, SOK
        """
        self._wait_rate_limit(is_post=True)
        path = "/v1/order"

        body_dict = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "executionType": execution_type,
            "timeInForce": time_in_force,
        }
        if price:
            body_dict["price"] = price
        if loss_cut_price:
            body_dict["losscutPrice"] = loss_cut_price

        body = json.dumps(body_dict)
        headers = self._get_headers("POST", path, body)

        response = self.session.post(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close_order(
        self,
        symbol: str,
        side: str,
        size: str,
        position_id: str,
        execution_type: str = "MARKET",
        price: Optional[str] = None,
        time_in_force: str = "FAK",
    ) -> Dict[str, Any]:
        """
        決済注文

        Args:
            symbol: 通貨ペア
            side: BUY or SELL (建玉と逆方向)
            size: 決済数量
            position_id: 建玉ID
            execution_type: MARKET, LIMIT, STOP
            price: 指値価格
            time_in_force: FAK, FAS, FOK, SOK
        """
        self._wait_rate_limit(is_post=True)
        path = "/v1/closeOrder"

        body_dict = {
            "symbol": symbol,
            "side": side,
            "size": size,
            "positionId": position_id,
            "executionType": execution_type,
            "timeInForce": time_in_force,
        }
        if price:
            body_dict["price"] = price

        body = json.dumps(body_dict)
        headers = self._get_headers("POST", path, body)

        response = self.session.post(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close_all_positions(
        self,
        symbol: str,
        side: str,
        execution_type: str = "MARKET",
        price: Optional[str] = None,
        time_in_force: str = "FAK",
    ) -> Dict[str, Any]:
        """
        一括決済注文

        Args:
            symbol: 通貨ペア
            side: BUY or SELL (建玉と逆方向)
            execution_type: MARKET, LIMIT, STOP
            price: 指値価格
            time_in_force: FAK, FAS, FOK, SOK
        """
        self._wait_rate_limit(is_post=True)
        path = "/v1/closeBulkOrder"

        body_dict = {
            "symbol": symbol,
            "side": side,
            "executionType": execution_type,
            "timeInForce": time_in_force,
        }
        if price:
            body_dict["price"] = price

        body = json.dumps(body_dict)
        headers = self._get_headers("POST", path, body)

        response = self.session.post(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        注文キャンセル

        Args:
            order_id: 注文ID
        """
        self._wait_rate_limit(is_post=True)
        path = "/v1/cancelOrder"

        body_dict = {"orderId": order_id}
        body = json.dumps(body_dict)
        headers = self._get_headers("POST", path, body)

        response = self.session.post(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            data=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_executions(
        self,
        order_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        約定情報取得

        Args:
            order_id: 注文ID
            execution_id: 約定ID
        """
        self._wait_rate_limit()
        path = "/v1/executions"
        params = {}
        if order_id:
            params["orderId"] = order_id
        if execution_id:
            params["executionId"] = execution_id

        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_latest_executions(
        self,
        symbol: str,
        count: int = 100,
    ) -> Dict[str, Any]:
        """
        最新約定一覧取得

        Args:
            symbol: 通貨ペア
            count: 取得件数 (最大100)
        """
        self._wait_rate_limit()
        path = "/v1/latestExecutions"
        params = {"symbol": symbol, "count": str(count)}
        headers = self._get_headers("GET", path)

        response = self.session.get(
            f"{self.PRIVATE_ENDPOINT}{path}",
            headers=headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


class GMOForexWebSocket:
    """GMOコイン WebSocket クライアント (Public)"""

    WS_ENDPOINT = "wss://forex-api.coin.z.com/ws/public/v1"

    def __init__(self):
        self.ws = None
        self._callbacks = {}

    def subscribe_ticker(self, symbol: str, callback) -> None:
        """
        ティッカー購読

        Args:
            symbol: 通貨ペア
            callback: コールバック関数
        """
        self._callbacks[f"ticker_{symbol}"] = callback

    def subscribe_kline(self, symbol: str, interval: str, callback) -> None:
        """
        ローソク足購読

        Args:
            symbol: 通貨ペア
            interval: 時間足
            callback: コールバック関数
        """
        self._callbacks[f"kline_{symbol}_{interval}"] = callback

    # WebSocket接続の実装は別途必要
    # ここではインターフェースのみ定義

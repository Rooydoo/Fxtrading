#!/usr/bin/env python3
"""
バックテストスクリプト
過去データでの戦略検証
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.builder import FeatureBuilder
from src.features.selector import FeatureSelector
from src.model.predictor import SignalPredictor
from src.trading.position import Side

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Backtester:
    """バックテストエンジン"""

    def __init__(
        self,
        initial_balance: float = 1000000,
        spread_pips: float = 1.5,
        slippage_pips: float = 0.5,
        commission_rate: float = 0.00002,
        risk_per_trade: float = 0.01,
    ):
        """
        Args:
            initial_balance: 初期資金
            spread_pips: スプレッド (pips)
            slippage_pips: スリッページ (pips)
            commission_rate: 手数料率
            risk_per_trade: 1トレードあたりリスク
        """
        self.initial_balance = initial_balance
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.commission_rate = commission_rate
        self.risk_per_trade = risk_per_trade

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        atr: pd.Series,
        symbol: str = "EUR_USD",
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.0,
    ) -> Dict[str, Any]:
        """
        バックテストを実行

        Args:
            df: OHLCVデータ
            signals: シグナル (1: ロング, -1: ショート, 0: ノーポジ)
            atr: ATR値
            symbol: 通貨ペア
            sl_atr_mult: SL ATR倍率
            tp_atr_mult: TP ATR倍率

        Returns:
            バックテスト結果
        """
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[Dict] = []

        position = None
        position_entry_price = 0
        position_side = None
        position_size = 0
        position_sl = 0
        position_tp = 0
        entry_idx = None

        pip_value = 100 if "JPY" in symbol else 10000
        spread = self.spread_pips / pip_value
        slippage = self.slippage_pips / pip_value

        for i in range(len(df)):
            current_price = df["close"].iloc[i]
            current_high = df["high"].iloc[i]
            current_low = df["low"].iloc[i]
            current_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]
            signal = signals.iloc[i] if i < len(signals) else 0

            # ポジションチェック
            if position is not None:
                # SL/TPチェック
                if position_side == Side.LONG:
                    if current_low <= position_sl:
                        # SLヒット
                        exit_price = position_sl - slippage
                        pnl = (exit_price - position_entry_price) * position_size
                        pnl -= abs(pnl) * self.commission_rate
                        balance += pnl
                        trades.append({
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "side": "LONG",
                            "entry_price": position_entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "pnl": pnl,
                            "exit_reason": "stop_loss",
                        })
                        position = None
                    elif current_high >= position_tp:
                        # TPヒット
                        exit_price = position_tp - slippage
                        pnl = (exit_price - position_entry_price) * position_size
                        pnl -= abs(pnl) * self.commission_rate
                        balance += pnl
                        trades.append({
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "side": "LONG",
                            "entry_price": position_entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "pnl": pnl,
                            "exit_reason": "take_profit",
                        })
                        position = None
                else:  # SHORT
                    if current_high >= position_sl:
                        exit_price = position_sl + slippage
                        pnl = (position_entry_price - exit_price) * position_size
                        pnl -= abs(pnl) * self.commission_rate
                        balance += pnl
                        trades.append({
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "side": "SHORT",
                            "entry_price": position_entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "pnl": pnl,
                            "exit_reason": "stop_loss",
                        })
                        position = None
                    elif current_low <= position_tp:
                        exit_price = position_tp + slippage
                        pnl = (position_entry_price - exit_price) * position_size
                        pnl -= abs(pnl) * self.commission_rate
                        balance += pnl
                        trades.append({
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "side": "SHORT",
                            "entry_price": position_entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "pnl": pnl,
                            "exit_reason": "take_profit",
                        })
                        position = None

            # 新規エントリー
            if position is None and signal != 0:
                if signal == 1:  # LONG
                    position_entry_price = current_price + spread + slippage
                    position_side = Side.LONG
                    position_sl = position_entry_price - current_atr * sl_atr_mult
                    position_tp = position_entry_price + current_atr * tp_atr_mult
                else:  # SHORT
                    position_entry_price = current_price - slippage
                    position_side = Side.SHORT
                    position_sl = position_entry_price + current_atr * sl_atr_mult
                    position_tp = position_entry_price - current_atr * tp_atr_mult

                # ポジションサイズ計算
                sl_distance = abs(position_entry_price - position_sl)
                risk_amount = balance * self.risk_per_trade
                position_size = risk_amount / sl_distance if sl_distance > 0 else 0

                position = "open"
                entry_idx = i

            equity_curve.append(balance)

        # 最終ポジションをクローズ
        if position is not None:
            exit_price = df["close"].iloc[-1]
            if position_side == Side.LONG:
                pnl = (exit_price - position_entry_price) * position_size
            else:
                pnl = (position_entry_price - exit_price) * position_size
            pnl -= abs(pnl) * self.commission_rate
            balance += pnl
            trades.append({
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "side": "LONG" if position_side == Side.LONG else "SHORT",
                "entry_price": position_entry_price,
                "exit_price": exit_price,
                "size": position_size,
                "pnl": pnl,
                "exit_reason": "end_of_data",
            })

        # 統計計算
        results = self._calculate_statistics(trades, equity_curve)
        results["trades"] = trades
        results["equity_curve"] = equity_curve

        return results

    def _calculate_statistics(
        self,
        trades: List[Dict],
        equity_curve: List[float],
    ) -> Dict[str, Any]:
        """統計を計算"""
        if not trades:
            return {
                "total_trades": 0,
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
            }

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # リターン
        total_return = (equity_curve[-1] - self.initial_balance) / self.initial_balance

        # シャープレシオ
        if len(pnls) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        else:
            sharpe = 0

        # 最大ドローダウン
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_drawdown = drawdown.max()

        # プロフィットファクター
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": sum(pnls),
            "total_return": total_return,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_balance": equity_curve[-1],
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="FX Trading Backtest")
    parser.add_argument("--symbol", default="EUR_USD", help="通貨ペア")
    parser.add_argument("--days", type=int, default=90, help="バックテスト日数")
    parser.add_argument("--initial-balance", type=float, default=1000000, help="初期資金")
    parser.add_argument("--model-path", help="モデルファイルパス")
    args = parser.parse_args()

    logger.info(f"Starting backtest for {args.symbol}")
    logger.info(f"Period: {args.days} days, Initial balance: ¥{args.initial_balance:,.0f}")

    # ここでは実際のデータ取得とモデルロードは省略
    # 実際の使用時はGMOForexClientでデータを取得し、モデルをロードする

    logger.info("Backtest completed")
    logger.info("Note: This is a template. Implement data fetching and model loading for actual use.")


if __name__ == "__main__":
    main()

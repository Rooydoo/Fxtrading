#!/usr/bin/env python3
"""
初回学習スクリプト
過去1〜2年分のデータを取得してモデルを学習
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.gmo_client import GMOForexClient
from src.data.fetcher import DataFetcher
from src.features.builder import FeatureBuilder
from src.features.selector import FeatureSelector
from src.model.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class InitialTrainer:
    """初回学習クラス"""

    def __init__(
        self,
        symbols: list = None,
        lookback_months: int = 12,
        output_dir: str = "models",
    ):
        """
        Args:
            symbols: 学習対象通貨ペア
            lookback_months: 学習データ期間 (月)
            output_dir: モデル出力ディレクトリ
        """
        self.symbols = symbols or ["EUR_USD", "USD_JPY"]
        self.lookback_months = lookback_months
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = GMOForexClient()
        self.fetcher = DataFetcher(self.client)
        self.feature_builder = FeatureBuilder("config/features.yaml")
        self.feature_selector = FeatureSelector()
        self.trainer = ModelTrainer()

    def fetch_historical_data(
        self,
        symbol: str,
        interval: str = "15m",
    ) -> pd.DataFrame:
        """
        過去データを取得

        Args:
            symbol: 通貨ペア
            interval: 時間軸

        Returns:
            OHLCVデータ
        """
        logger.info(f"Fetching {self.lookback_months} months of {symbol} data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_months * 30)

        all_data = []
        current_date = start_date

        while current_date < end_date:
            batch_end = min(current_date + timedelta(days=30), end_date)

            try:
                df = self.fetcher.fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start_date=current_date.strftime("%Y%m%d"),
                    end_date=batch_end.strftime("%Y%m%d"),
                )

                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  {current_date.date()} - {batch_end.date()}: {len(df)} rows")

            except Exception as e:
                logger.warning(f"  Failed to fetch {current_date.date()}: {e}")

            current_date = batch_end

        if not all_data:
            logger.error(f"No data fetched for {symbol}")
            return pd.DataFrame()

        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        logger.info(f"Total {symbol} data: {len(combined)} rows ({combined.index.min()} to {combined.index.max()})")

        return combined

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> tuple:
        """
        学習データを準備

        Args:
            df: OHLCVデータ
            symbol: 通貨ペア

        Returns:
            (X, y, feature_names)
        """
        logger.info("Building features...")

        # 上位時間軸データ
        df_1h = df.resample("1H").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }).dropna()

        # 特徴量生成
        df_features = self.feature_builder.build_all_features(df, df_1h)

        # ターゲット生成 (次の足の方向)
        target = self.trainer.prepare_target(
            df_features,
            target_col="close",
            lookahead=1,
            threshold=0.0,
        )

        # NaN除去
        feature_columns = self.feature_builder.get_feature_names(df_features)
        valid_mask = df_features[feature_columns].notna().all(axis=1) & target.notna()

        X = df_features.loc[valid_mask, feature_columns]
        y = target.loc[valid_mask]

        logger.info(f"Training samples: {len(X)}, Features: {len(feature_columns)}")

        return X, y, feature_columns

    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: list,
        tune_hyperparams: bool = False,
        n_trials: int = 50,
    ) -> dict:
        """
        モデルを学習

        Args:
            X: 特徴量
            y: ターゲット
            feature_names: 特徴量名
            tune_hyperparams: ハイパーパラメータチューニング
            n_trials: チューニング試行数

        Returns:
            学習結果
        """
        results = {}

        # ハイパーパラメータチューニング
        if tune_hyperparams:
            logger.info(f"Running hyperparameter tuning ({n_trials} trials)...")
            best_params = self.trainer.hyperparameter_tuning(X, y, n_trials=n_trials)
            results["best_params"] = best_params

        # ウォークフォワード検証
        logger.info("Running walk-forward validation...")
        wf_results = self.trainer.walk_forward_validation(
            df=X,
            feature_columns=feature_names,
            target=y,
            n_splits=5,
            test_size=0.2,
        )
        results["walk_forward"] = wf_results

        # 全データで最終モデルを学習
        logger.info("Training final model on all data...")
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]

        self.trainer.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # 評価
        eval_metrics = self.trainer.evaluate_model(X_val, y_val)
        results["evaluation"] = eval_metrics

        # 特徴量重要度
        importance = self.trainer.model.feature_importance(importance_type="gain")
        results["feature_importance"] = dict(zip(feature_names, importance))

        return results

    def run(
        self,
        tune_hyperparams: bool = False,
        n_trials: int = 50,
    ) -> dict:
        """
        初回学習を実行

        Args:
            tune_hyperparams: ハイパーパラメータチューニング
            n_trials: チューニング試行数

        Returns:
            全体結果
        """
        all_results = {}
        combined_X = []
        combined_y = []
        feature_names = None

        for symbol in self.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*50}")

            # データ取得
            df = self.fetch_historical_data(symbol)
            if df.empty:
                continue

            # データ保存
            data_file = self.output_dir / f"historical_{symbol}.parquet"
            df.to_parquet(data_file)
            logger.info(f"Data saved to {data_file}")

            # 学習データ準備
            X, y, feat_names = self.prepare_training_data(df, symbol)

            if len(X) < 1000:
                logger.warning(f"Not enough data for {symbol}: {len(X)} samples")
                continue

            combined_X.append(X)
            combined_y.append(y)
            feature_names = feat_names

            all_results[symbol] = {
                "samples": len(X),
                "date_range": f"{X.index.min()} - {X.index.max()}",
            }

        if not combined_X:
            logger.error("No valid training data")
            return {}

        # 全データを結合
        X_all = pd.concat(combined_X)
        y_all = pd.concat(combined_y)

        logger.info(f"\n{'='*50}")
        logger.info(f"Combined training data: {len(X_all)} samples")
        logger.info(f"{'='*50}")

        # モデル学習
        train_results = self.train_model(
            X_all, y_all, feature_names,
            tune_hyperparams=tune_hyperparams,
            n_trials=n_trials,
        )

        all_results["training"] = train_results

        # モデル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.output_dir / f"model_initial_{timestamp}.pkl"
        self.trainer.save_model(str(model_path))

        # current_model.pklとしてもコピー
        current_path = self.output_dir / "current_model.pkl"
        self.trainer.save_model(str(current_path))

        all_results["model_path"] = str(model_path)
        all_results["current_model_path"] = str(current_path)

        # 結果保存
        results_path = self.output_dir / f"training_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\n{'='*50}")
        logger.info("Training completed!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'='*50}")

        # サマリー表示
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: dict) -> None:
        """結果サマリーを表示"""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        if "training" in results:
            tr = results["training"]

            if "walk_forward" in tr:
                wf = tr["walk_forward"]
                print(f"\nWalk-Forward Validation:")
                print(f"  Mean Accuracy: {wf.get('mean_accuracy', 0):.4f}")
                print(f"  Std Accuracy:  {wf.get('std_accuracy', 0):.4f}")
                print(f"  Direction Acc: {wf.get('direction_accuracy', 0):.4f}")

            if "evaluation" in tr:
                ev = tr["evaluation"]
                print(f"\nFinal Model Evaluation:")
                print(f"  Accuracy:  {ev.get('accuracy', 0):.4f}")
                print(f"  Precision: {ev.get('precision', 0):.4f}")
                print(f"  Recall:    {ev.get('recall', 0):.4f}")
                print(f"  F1 Score:  {ev.get('f1_score', 0):.4f}")

            if "feature_importance" in tr:
                print(f"\nTop 10 Important Features:")
                fi = tr["feature_importance"]
                sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feat, imp) in enumerate(sorted_fi, 1):
                    print(f"  {i:2d}. {feat}: {imp:.4f}")

        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Initial Model Training")
    parser.add_argument("--symbols", nargs="+", default=["EUR_USD", "USD_JPY"],
                        help="通貨ペア")
    parser.add_argument("--months", type=int, default=12,
                        help="学習データ期間 (月)")
    parser.add_argument("--tune", action="store_true",
                        help="ハイパーパラメータチューニング実行")
    parser.add_argument("--trials", type=int, default=50,
                        help="チューニング試行数")
    parser.add_argument("--output", default="models",
                        help="出力ディレクトリ")
    args = parser.parse_args()

    # ワーキングディレクトリ
    os.chdir(Path(__file__).parent.parent)

    trainer = InitialTrainer(
        symbols=args.symbols,
        lookback_months=args.months,
        output_dir=args.output,
    )

    results = trainer.run(
        tune_hyperparams=args.tune,
        n_trials=args.trials,
    )

    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()

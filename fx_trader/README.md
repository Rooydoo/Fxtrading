# FX Machine Learning Trading System

GMOコイン外国為替FX APIを使用した機械学習ベースの自動売買システム。

## 機能

- LightGBMによるシグナル生成
- リスク管理（ロング/ショート別パラメータ）
- Telegram通知（売買・日次/週次/月次レポート）
- ペーパートレード機能
- 特徴量の動的ON/OFF（YAML管理）
- 自動再学習機能
- VPS上でのSystemd常時稼働

## 必要条件

- Python 3.9+
- GMOコイン外国為替FX APIアカウント

## インストール

```bash
# リポジトリをクローン
git clone <repository_url>
cd fx_trader

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

## 設定

1. 環境変数を設定:

```bash
cp .env.example .env
# .envを編集してAPIキーを設定
```

2. 設定ファイルを編集:

- `config/settings.yaml` - システム全体設定
- `config/features.yaml` - 特徴量設定
- `config/risk_params.yaml` - リスク管理パラメータ
- `config/retraining.yaml` - 再学習トリガー設定

## 使い方

### ペーパートレード

```bash
python main.py --mode paper --model models/model.pkl
```

### 本番稼働

```bash
python main.py --mode live --model models/model.pkl
```

### Systemdでの常時稼働

```bash
# サービスファイルをコピー
sudo cp fx_trader.service /etc/systemd/system/

# サービスを有効化
sudo systemctl enable fx_trader
sudo systemctl start fx_trader

# ステータス確認
sudo systemctl status fx_trader
```

## バックテスト

```bash
python scripts/backtest.py --symbol EUR_USD --days 90
```

## モデル再学習

```bash
python scripts/retrain.py --symbol EUR_USD --tune --n-trials 50
```

## ディレクトリ構造

```
fx_trader/
├── config/           # 設定ファイル
├── src/
│   ├── data/         # データ取得
│   ├── features/     # 特徴量生成
│   ├── model/        # 機械学習モデル
│   ├── trading/      # 売買実行
│   ├── notification/ # 通知
│   ├── monitoring/   # モニタリング
│   └── core/         # コア機能
├── scripts/          # スクリプト
├── data/             # データ保存
├── models/           # モデル保存
├── logs/             # ログ
└── main.py           # エントリーポイント
```

## 注意事項

- 本システムは投資助言を行うものではありません
- 実際の取引は自己責任で行ってください
- 十分なペーパートレードテストを行ってから本番稼働してください

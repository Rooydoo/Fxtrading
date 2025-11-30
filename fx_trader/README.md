# FX Machine Learning Trading System

GMOコイン外国為替FX APIを使用した機械学習ベースの自動売買システム。

## 機能

- **LightGBM**によるシグナル生成（15分足）
- **リスク管理**（ロング/ショート別パラメータ、ATRベースSL/TP）
- **トレーリングストップ**（利益を最大化）
- **Telegram通知＆コマンド**（売買通知、ステータス確認、緊急決済）
- **ペーパートレード機能**（仮想予算でのテスト）
- **経済指標カレンダー**（重要指標発表時は取引回避）
- **ポジション復旧**（システム再起動時に自動復元）
- **データキャッシュ**（API負荷軽減）
- **特徴量の動的ON/OFF**（YAML管理）
- **自動再学習機能**

---

## クイックスタート

### 1. 環境構築

```bash
cd fx_trader

# 仮想環境を作成
python3 -m venv venv
source venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. 環境変数設定

```bash
cp .env.example .env
nano .env  # APIキーとTelegramトークンを設定
```

### 3. 初回モデル学習

```bash
# 過去12ヶ月のデータでモデル学習
python scripts/initial_training.py --months 12
```

### 4. ペーパートレード開始

```bash
python main.py --mode paper
```

---

## VPSへのデプロイ

### 前提条件

- Ubuntu 20.04+ / Debian 11+
- Python 3.9+
- 最低1GB RAM推奨

### 手順

#### 1. システム準備

```bash
# パッケージ更新
sudo apt update && sudo apt upgrade -y

# Python環境インストール
sudo apt install -y python3 python3-pip python3-venv git

# プロジェクトディレクトリ作成
sudo mkdir -p /opt/fx_trader
sudo chown $USER:$USER /opt/fx_trader
```

#### 2. プロジェクトセットアップ

```bash
cd /opt/fx_trader

# リポジトリをクローン（またはファイルをアップロード）
git clone <repository_url> .

# 仮想環境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 環境変数
cp .env.example .env
nano .env  # APIキーを設定

# ディレクトリ作成
mkdir -p data logs models
```

#### 3. 初回モデル学習

```bash
source venv/bin/activate
python scripts/initial_training.py --months 12 --tune --trials 50
```

#### 4. Systemdサービス設定

```bash
# サービスファイルを編集（パスを確認）
sudo nano /etc/systemd/system/fx_trader.service
```

```ini
[Unit]
Description=FX ML Trading System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/fx_trader
Environment="PATH=/opt/fx_trader/venv/bin"
EnvironmentFile=/opt/fx_trader/.env
ExecStart=/opt/fx_trader/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# サービスを有効化
sudo systemctl daemon-reload
sudo systemctl enable fx_trader
sudo systemctl start fx_trader

# ステータス確認
sudo systemctl status fx_trader
```

#### 5. ログ確認

```bash
# Systemdログ
sudo journalctl -u fx_trader -f

# アプリケーションログ
tail -f /opt/fx_trader/logs/fx_trader.log
```

---

## Telegramコマンド

| コマンド | 説明 |
|---------|------|
| `/status` | システムステータス表示 |
| `/balance` | 残高と損益を表示 |
| `/positions` | オープンポジション一覧 |
| `/report [日数]` | パフォーマンスレポート |
| `/stats` | 取引統計 |
| `/pause` | 取引を一時停止 |
| `/resume` | 取引を再開 |
| `/closeall confirm` | 全ポジション緊急決済 |
| `/risk` | リスク設定表示 |
| `/help` | ヘルプ表示 |

---

## 設定ファイル

| ファイル | 内容 |
|---------|------|
| `config/settings.yaml` | システム全体設定、通貨ペア、API設定 |
| `config/features.yaml` | 特徴量ON/OFF設定 |
| `config/risk_params.yaml` | リスク管理、SL/TP、トレーリングストップ |
| `config/retraining.yaml` | 自動再学習トリガー設定 |

---

## ディレクトリ構造

```
fx_trader/
├── config/               # 設定ファイル
├── src/
│   ├── data/             # API、データ取得、キャッシュ、カレンダー
│   ├── features/         # 特徴量生成
│   ├── model/            # LightGBMモデル
│   ├── trading/          # 売買実行、リスク管理、トレーリングストップ
│   ├── notification/     # Telegram通知、コマンド
│   ├── monitoring/       # パフォーマンス監視
│   └── core/             # モード管理、スケジューラー
├── scripts/              # 学習・検証スクリプト
│   ├── initial_training.py  # 初回学習
│   ├── backtest.py          # バックテスト
│   ├── walk_forward.py      # ウォークフォワード検証
│   └── retrain.py           # 再学習
├── data/                 # データ保存（自動生成）
├── models/               # モデル保存
├── logs/                 # ログファイル
├── main.py               # エントリーポイント
├── requirements.txt      # 依存パッケージ
└── fx_trader.service     # Systemdサービスファイル
```

---

## トラブルシューティング

### システムが起動しない

```bash
# ログを確認
sudo journalctl -u fx_trader -n 100

# 手動で実行してエラーを確認
cd /opt/fx_trader
source venv/bin/activate
python main.py --once
```

### Telegram通知が来ない

1. `.env`の`TELEGRAM_BOT_TOKEN`と`TELEGRAM_CHAT_ID`を確認
2. ボットとのチャットで`/start`を送信済みか確認
3. ネットワーク接続を確認

### API接続エラー

1. GMO Coinの API管理画面でAPIが有効か確認
2. IPアドレス制限を確認（VPSのIPを許可）
3. API権限（建玉参照、注文、口座情報）を確認

---

## 注意事項

- 本システムは投資助言を行うものではありません
- 実際の取引は自己責任で行ってください
- **最低3ヶ月のペーパートレード検証**を推奨します
- 本番運用前に必ずリスクパラメータを確認してください

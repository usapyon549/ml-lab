# ml-lab

機械学習・MLOps・LLM 周辺技術の軽量実験リポジトリ。

portfolio リポジトリとは別に、  
「技術を試す場所」として運用しています。

---

# Purpose

- MLOps 技術に触れる
- Docker / AWS / MLflow などの周辺技術を試す
- 学習ログ・試行錯誤を残す

完成度の高い portfolio というより、

- 実験
- 検証
- 学習記録
- 試作

を目的とした repository です。

---

# Projects

## 1. electricity-demand-forecast

東京電力の公開データを用いた電力需要予測。

### What I tried

- LightGBM による時系列予測
- lag / rolling 特徴量
- sin / cos による周期特徴量
- SHAP によるモデル解釈
- MLflow による experiment tracking

### Current status

- baseline → feature engineering により精度改善
- MLflow による実験管理を導入
- Docker container で推論実行を確認

[電力需要予測のリンク](/projects/electricity-demand-forecast/README.md)

## 2. demand-forecast-sagemaker-deploy

`electricity-demand-forecast` で作成した LightGBM モデルを、  
Amazon SageMaker のサーバーレスエンドポイントへデプロイ。

### What I tried
- SageMaker custom inference container の構築
- sagemaker-inference を用いた推論サーバー実装
- Apple Silicon 環境からの amd64 build
- ECR へのコンテナ push
- SageMaker Serverless Endpoint
- boto3 invoke_endpoint() による疎通確認
- CloudWatch を用いたコンテナ障害調査

### Current status
- custom inference container による deploy 成功
- SageMaker endpoint が InService で稼働確認済み
- notebook から推論レスポンス取得済み
- ハマりどころ・対応内容を README に整理

[電力需要予測モデルのSageMakerデプロイのリンク](/projects/demand-forecast-sagemaker-deploy/README.md)

## 3. banking-marketing(実施予定)

Huging Faceの公開データセットをもとに、銀行のマーケティング成否の予測。

### What I WILL try

- マーケティング成否の２値分類モデル
- 複数モデルの比較(lightGBM, SVM, LogisticRegression)
- 不均衡データの予測
- 正規化を含めた前処理
- 簡易アンサンブル学習の実施(XGboostを用いて、lightGBM、SVM, LogisticRegressionの３モデルの予測結果をstacking)

---

# Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- LightGBM
- MLflow
- SHAP
- Docker
- AWS SageMaker
- AWS ECR
- boto3

---

# Notes

- Notebook ベースで軽量に実験
- 試行錯誤の過程も一部含みます
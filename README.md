# ml-lab

機械学習・MLOps・LLM 周辺技術の軽量実験リポジトリ。

portfolio リポジトリとは別に、  
「技術を試す場所」として運用しています。

[portfolioはこちら](https://github.com/usapyon549/portfolio)

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

- ベースラインから特徴量エンジニアリングにより精度を改善
- MLflow による実験管理フローを構築
- Docker コンテナ上で推論が実行できることを確認

[電力需要予測のリンク](/projects/001-electricity-demand-forecast/README.md)

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

[電力需要予測モデルのSageMakerデプロイのリンク](/projects/002-demand-forecast-sagemaker-deploy/README.md)

## 3. banking-marketing

Banking Marketing Dataset を用いた定期預金契約有無の分類予測。

クラス不均衡データを対象に、複数モデルの比較や前処理の影響を検証。

### What I tried
- LightGBM
- Logistic Regression
- Support Vector Machine (SVM)
- class_weight を用いた不均衡データ対応
- StandardScaler による特徴量スケーリング
- Recall / F1 を用いたモデル評価
- MLflow による experiment tracking
- StackingClassifier を用いたアンサンブル学習

### Current status
- LightGBM・Logistic Regression・SVM の比較完了
- 不均衡データにおける Accuracy の限界を確認
- class_weight による Recall 改善を確認
- SVM における標準化の重要性を確認
- StackingClassifier を検証したが、単体モデルを上回る性能向上は確認できなかった

[banking-marketingのリンク](/projects/003-banking-marketing/README.md)


## 4. portfolio-rag-assistant

ポートフォリオ情報を自然言語で検索できる RAG アシスタント。

SentenceTransformer、ChromaDB、Gemini API を組み合わせ、  
ベクトル検索と生成AIを利用した検索システムを構築。

### What I tried

* SentenceTransformer
* Embedding
* cosine similarity
* ChromaDB
* Chunking
* Top-K Retrieval
* Gemini API
* Prompt Engineering
* Streamlit
* RAG (Retrieval-Augmented Generation)

### Current status

* Embedding の仕組みを検証
* cosine similarity による類似度計算を実装
* ChromaDB によるベクトル検索を実装
* Chunking による文書分割を実装
* Top-K Retrieval を実装
* Gemini API を利用した回答生成を実装
* Prompt Engineering により回答品質を改善
* Streamlit による Web UI を構築
* Streamlit Cache により Embedding モデルの再ロードを防止
* Portfolio RAG Assistant として動作確認完了

[portfolio-rag-assistant のリンク](/projects/004-portfolio-rag-assistant/README.md)


---

# Tech Stack

### Machine Learning

* Python
* pandas
* numpy
* scikit-learn
* LightGBM
* XGBoost
* MLflow
* SHAP
* Support Vector Machine(SVM)

### LLM / RAG

* Gemini API
* Sentence Transformers
* ChromaDB
* Embedding
* Vector Search
* Retrieval-Augmented Generation (RAG)
* Prompt Engineering

### MLOps

* Docker
* AWS ECR
* AWS SageMaker
* boto3

### Application

* Streamlit
* FastAPI

### Machine Learning Topics

* Time Series Forecasting
* Binary Classification
* Class Imbalance Handling
* Feature Engineering
* Feature Scaling
* Ensemble Learning (Stacking)
* Threshold Optimization
* Model Interpretation
* Experiment Tracking



---

# Notes

- Notebook ベースで軽量に実験
- 試行錯誤の過程も一部含みます
- 完成品よりも、技術検証や学習ログを重視したリポジトリとして運用しています
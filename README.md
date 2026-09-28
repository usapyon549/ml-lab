# ml-lab

機械学習・MLOps・LLM 周辺技術の軽量実験リポジトリです。

portfolio リポジトリとは別に、
「技術を試す場所」として運用しています。

[portfolioはこちら](https://github.com/usapyon549/portfolio)

# Projects

## 1. electricity-demand-forecast

東京電力の公開データを用いた電力需要予測。

### What I tried

* LightGBM による時系列予測
* lag / rolling 特徴量
* sin / cos による周期特徴量
* SHAP によるモデル解釈
* MLflow による experiment tracking
* 時刻・気温を中心としたベースラインから、特徴量追加による性能変化を比較
* 特徴量構成の違いによるモデル性能・特徴量重要度の変化を検証
* Docker コンテナ上での推論実行

[電力需要予測のリンク](/projects/001-electricity-demand-forecast/README.md)

## 2. demand-forecast-sagemaker-deploy

`electricity-demand-forecast` で作成した LightGBM モデルを、
Amazon SageMaker のサーバーレスエンドポイントへデプロイ。

### What I tried

* SageMaker custom inference container の構築
* sagemaker-inference を用いた推論サーバー実装
* Apple Silicon 環境からの amd64 build
* ECR へのコンテナ push
* SageMaker Serverless Endpoint
* boto3 invoke_endpoint() による推論リクエスト
* CloudWatch を用いたコンテナ障害調査
* custom inference container によるモデルデプロイから、endpoint の InService 確認までの一連の流れを検証

[電力需要予測モデルのSageMakerデプロイのリンク](/projects/002-demand-forecast-sagemaker-deploy/README.md)

## 3. banking-marketing

Banking Marketing Dataset を用いた定期預金契約有無の分類予測。

クラス不均衡データを対象に、複数モデルの比較や前処理の影響を検証。

### What I tried

* LightGBM
* Logistic Regression
* Support Vector Machine (SVM)
* class_weight を用いた不均衡データ対応
* StandardScaler による特徴量スケーリング
* Recall / F1 を用いたモデル評価
* MLflow による experiment tracking
* StackingClassifier を用いたアンサンブル学習
* モデルごとの性能と、不均衡データへの前処理・重み付けの影響を比較
* SVM における特徴量スケーリングの影響を検証
* StackingClassifier によるアンサンブル学習の効果を検証

[banking-marketingのリンク](/projects/003-banking-marketing/README.md)

## 4. portfolio-rag-assistant

ポートフォリオ情報を自然言語で検索できる RAG アシスタント。

SentenceTransformer、ChromaDB、Gemini API を組み合わせ、
ベクトル検索と生成AIを利用した検索システムを構築。

### What I tried

* SentenceTransformer による Embedding
* cosine similarity による類似度計算
* ChromaDB によるベクトル検索
* Chunking
* Top-K Retrieval
* Gemini API による回答生成
* Prompt Engineering
* Streamlit による Web UI
* Chunking や Top-K Retrieval の設定による検索結果の変化を検証
* Prompt Engineering による回答品質の改善
* Streamlit Cache による Embedding モデルの再ロード抑制

[portfolio-rag-assistant のリンク](/projects/004-portfolio-rag-assistant/README.md)

## 5. forest-cover-type-classification

森林の土地被覆タイプを予測する分類問題を用いて、
**学習データ量や特徴量数がモデル性能に与える影響を検証。**

### What I tried

* LightGBM による多クラス分類
* 特徴量数を変更したモデル性能の比較
* 学習データ量を変更したモデル性能の比較
* ランダムな特徴量選択による複数回実験
* Precision / Recall / F1 によるクラス別評価
* Macro F1 / Weighted F1 による全体評価
* 特徴量数と学習データ量を組み合わせた実験
* 10 / 20 / 30 / 40 特徴量による性能変化を比較
* ランダムな特徴量選択による結果のばらつきを検証
* 54特徴量を固定して学習データ量を変更し、性能への影響を比較
* Accuracy と Macro F1 / クラス別 F1 を比較し、データ量によるクラスごとの性能変化を検証

[forest-cover-type-classification のリンク](/projects/005-forest-cover-type-classification/README.md)

## 6. portfolio-knowledge-graph-rag

ポートフォリオのREADME群を対象に、
**通常のRAGとKnowledge Graph RAGで、文書間の関係性を考慮した検索・回答にどのような違いが生じるかを検証。**

### What I tried

* LightRAG による Knowledge Graph RAG の構築
* Ollama / BGE-M3 によるローカルEmbedding
* Gemini API によるエンティティ・関係抽出および回答生成
* Hybrid Search による Knowledge Graph RAG の検索
* 通常のRAGとの回答内容の比較（004-portfolio-rag-assistantとの比較）
* Knowledge Graph の可視化
* 複数の質問パターンによる回答傾向の比較
* プロジェクト横断の関連性や共通する技術要素について、通常のRAGとの違いを検証
* 元文書の記述内容がKnowledge Graphの構築や検索・回答に与える影響を確認

[knowledge-graph-rag のリンク](./projects/006-portfolio-knowledge-graph-rag/README.md)

## 7. toy-llm-adversarial-data

小規模なTransformerをPyTorchで自作し、
**正しい計算ルール・偽の計算ルール・ルールのないランダムなデータを混ぜたとき、モデルの出力がどのように変化するかを検証。**

### What I tried

* PyTorchによる小規模Transformerの実装
* Character-level Tokenization
* Token Embedding / Position Embedding
* Causal MaskによるDecoder-only型に近い構成
* Paddingおよび`ignore_index`を利用したLoss計算
* 正しい計算ルール・偽の計算ルール・ランダムデータを混ぜた学習
* データ比率によるモデルの出力傾向の比較
* 正しいルールと偽のルールの比率による出力傾向の変化を検証
* ランダムデータを混ぜた場合の出力への影響を比較
* 系列長の違いが学習結果に影響する可能性について考察
* 学習したルールの獲得と、データの記憶・一般化を区別できていない点を整理

[toy-llm-adversarial-data のリンク](./projects/007-toy-llm-adversarial-data/README.md)

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
* PyTorch

### LLM / RAG

* Gemini API
* Sentence Transformers
* ChromaDB
* Embedding
* Vector Search
* Retrieval-Augmented Generation (RAG)
* Prompt Engineering
* Knowledge-graph-rag
* LightRAG
* Ollama
* Transformer
* Tokenization
* Position Embedding

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
* Binary / Multiclass Classification
* Class Imbalance Handling
* Feature Engineering
* Feature Scaling
* Ensemble Learning (Stacking)
* Threshold Optimization
* Model Interpretation
* Experiment Tracking
* Feature Selection
* Data Size Analysis
* Transformer / Attention
* Sequence Modeling

---

# Notes

* Notebook ベースで軽量に実験
* 試行錯誤の過程も一部含みます
* 完成品よりも、技術検証や学習ログを重視したリポジトリとして運用しています

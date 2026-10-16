# ml-lab

機械学習・MLOps・LLM 周辺技術の軽量実験リポジトリです。

`portfolio` リポジトリとは別に、技術検証や学習ログを蓄積する「技術を試す場所」として運用しています。

[portfolioはこちら](https://github.com/usapyon549/portfolio)

## Projects

### 1. electricity-demand-forecast

東京電力エリアの公開データを用いた電力需要予測。

**What I tried**
- LightGBM による時系列予測
- Lag / Rolling 特徴量、sin / cos による周期特徴量
- SHAP によるモデル解釈
- MLflow による実験管理
- 特徴量構成によるモデル性能・特徴量重要度の変化を比較
- Docker コンテナ上での推論実行

[electricity-demand-forecastへのリンク](./projects/001-electricity-demand-forecast/README.md)

### 2. demand-forecast-sagemaker-deploy

`electricity-demand-forecast` で作成した LightGBM モデルを、Amazon SageMaker の Serverless Endpoint へデプロイ。

**What I tried**
- SageMaker Custom Inference Container の構築
- `sagemaker-inference` を用いた推論サーバーの実装
- Apple Silicon 環境からの amd64 コンテナイメージのビルド
- Amazon ECR へのイメージ登録
- SageMaker Serverless Endpoint の作成・デプロイ
- `boto3` による推論リクエスト
- CloudWatch を用いたコンテナ障害の調査
- モデルのコンテナ化から Endpoint の `InService` 確認までの一連の流れを検証

[demand-forecast-sagemaker-deployへのリンク](./projects/002-demand-forecast-sagemaker-deploy/README.md)

### 3. banking-marketing

Banking Marketing Dataset を用いた定期預金契約有無の分類予測。

クラス不均衡データを対象に、複数モデルの比較や前処理の影響を検証。

**What I tried**
- LightGBM / Logistic Regression / SVM による分類
- `class_weight` によるクラス不均衡への対応
- `StandardScaler` による特徴量スケーリング
- Recall / F1 を重視したモデル評価
- MLflow による実験管理
- `StackingClassifier` によるアンサンブル学習
- モデルごとの性能と、不均衡データへの重み付け・スケーリングの影響を比較

[banking-marketingへのリンク](./projects/003-banking-marketing/README.md)

### 4. portfolio-rag-assistant

ポートフォリオ情報を自然言語で検索できる RAG アシスタント。

Sentence Transformers、ChromaDB、Gemini API を組み合わせ、ベクトル検索と生成 AI を利用した検索システムを構築。

**What I tried**
- Sentence Transformers による Embedding
- ChromaDB によるベクトル検索
- Chunking / Top-K Retrieval
- Cosine Similarity による類似度計算
- Gemini API による回答生成
- Prompt Engineering による回答品質の改善
- Streamlit による Web UI
- Chunking / Top-K などの検索設定による検索結果の変化を検証
- Streamlit Cache による Embedding モデルの再ロード抑制

[portfolio-rag-assistantへのリンク](./projects/004-portfolio-rag-assistant/README.md)

### 5. forest-cover-type-classification

森林の土地被覆タイプを予測する分類問題を用いて、**学習データ量や特徴量数がモデル性能に与える影響を検証。**

**What I tried**
- LightGBM による多クラス分類
- 特徴量数・学習データ量を変更したモデル性能の比較
- ランダムな特徴量選択による複数回実験
- Precision / Recall / F1 によるクラス別評価
- Macro F1 / Weighted F1 による全体評価
- Accuracy とクラス別 F1 を用いた、データ量による性能変化の分析
- 特徴量数と学習データ量を組み合わせた実験
- ランダムな特徴量選択による結果のばらつきを検証

[forest-cover-type-classificationへのリンク](./projects/005-forest-cover-type-classification/README.md)

### 6. portfolio-knowledge-graph-rag

ポートフォリオの README 群を対象に、通常の RAG と Knowledge Graph RAG で、文書間の関係性を考慮した検索・回答にどのような違いが生じるかを検証。

**What I tried**
- LightRAG による Knowledge Graph RAG の構築
- Ollama / BGE-M3 によるローカル Embedding
- Gemini API によるエンティティ・関係抽出および回答生成
- Hybrid Search による検索
- 通常の RAG との回答内容の比較
- Knowledge Graph の可視化
- 複数の質問パターンによる回答傾向の比較
- プロジェクト横断の関連性や共通する技術要素を検証
- 元文書の記述内容が Graph 構築・検索・回答に与える影響を確認

[portfolio-knowledge-graph-ragへのリンク](./projects/006-portfolio-knowledge-graph-rag/README.md)

### 7. toy-llm-adversarial-data

PyTorch で小規模な Transformer を実装し、正しい計算ルール・偽の計算ルール・ルールのないランダムデータを混ぜたとき、モデルの出力がどのように変化するかを検証。

**What I tried**
- PyTorch による小規模 Transformer の実装
- Character-level Tokenization
- Token Embedding / Position Embedding
- Causal Mask を用いた Decoder-only 型に近い構成
- Padding と `ignore_index` を用いた Loss 計算
- 学習データの種類・比率による出力傾向の比較
- 系列長の違いが学習結果に影響する可能性について考察
- 学習したルールの獲得と、データの記憶・一般化を区別できていない点を整理

[toy-llm-adversarial-dataへのリンク](./projects/007-toy-llm-adversarial-data/README.md)

### 8. character-chat-llm

小規模な日本語対応 LLM を用いたキャラクターチャット実験。

LoRA による追加学習と Prompt Engineering / Few-shot の組み合わせを比較し、キャラクターらしい文章表現と回答品質への影響を検証。

**What I tried**
- Qwen2.5-0.5B-Instruct を用いたキャラクターチャット
- unsloth/PEFT による LoRA / QLoRA の実装
- キャラクターごとの発話データを用いた追加学習
- Base Model / LoRA / Few-shot・Prompt Engineering / LoRA + Few-shot の比較
- Gemini API を用いた参照条件との比較
- キャラクターらしさ・文章の自然さ・質問への適切さの観点から出力を定性的に評価
- キャラクター表現と回答品質の両立に関する課題を整理

[character-chat-llmへのリンク](./projects/008-character-chat-llm/README.md)

---

## Tech Stack

### Machine Learning
- Python
- pandas / NumPy
- scikit-learn
- LightGBM / XGBoost
- PyTorch
- MLflow
- SHAP

### LLM / RAG
- Hugging Face Transformers
- PEFT / LoRA / QLoRA
- Gemini API
- Sentence Transformers
- ChromaDB
- LightRAG
- Ollama
- Embedding / Vector Search
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering / Few-shot Learning
- Transformer / Tokenization / Position Embedding

### MLOps / Cloud
- Docker
- AWS ECR
- Amazon SageMaker
- boto3
- CloudWatch

### Application
- Streamlit
- FastAPI

## Topics

- Time Series Forecasting
- Binary / Multiclass Classification
- Class Imbalance Handling
- Feature Engineering / Feature Scaling / Feature Selection
- Ensemble Learning
- Model Interpretation
- Experiment Tracking
- Data Size Analysis
- Transformer / Attention / Sequence Modeling
- Parameter-Efficient Fine-Tuning (PEFT)
- RAG / Knowledge Graph RAG
- Model Deployment / Containerization

## Notes

- Notebook ベースの軽量な実験を中心に運用
- 技術検証の過程や試行錯誤も一部含む
- 完成品の公開よりも、技術検証・学習ログ・実験結果の整理を重視
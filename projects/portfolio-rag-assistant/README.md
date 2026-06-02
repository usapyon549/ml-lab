# portfolio-rag-assistant

ポートフォリオおよび機械学習プロジェクトを検索対象とした RAG (Retrieval-Augmented Generation) アプリケーション。

potfolioで初めてLLMを触った状態から、本プロジェクトではEmbedding によるベクトル検索と   
Gemini API を組み合わせ、自然言語によるプロジェクト検索・質問応答を実装した。

## What I tried

* Sentence Transformers による Embedding
* ChromaDB による Vector Store
* Cosine Similarity を用いたベクトル検索
* Chunking
* Metadata 管理
* Retrieval-Augmented Generation (RAG)
* Gemini API
* Prompt Engineering

---

## Architecture

```text
User Query
    ↓
Embedding
    ↓
Vector Search (ChromaDB)
    ↓
Top-K Retrieval
    ↓
Context Generation
    ↓
Gemini API
    ↓
Answer
```

---

## Data Source

検索対象として、自作の機械学習プロジェクトドキュメントを利用。

対象プロジェクト例：

* pv-forecasting
* electricity-demand-forecast
* demand-forecast-sagemaker-deploy
* weather-classification
* banking-marketing

各プロジェクトの README を Markdown 化し、RAG の知識ベースとして利用した。

---

## What I Learned

### Embedding

Sentence Transformer を利用し、文章を意味ベクトルへ変換。

単語一致ではなく、意味的に近い文章同士が近いベクトル空間へ配置されることを確認した。

### Vector Search

ChromaDB を利用してベクトル検索を実装。

Cosine Similarity により、ユーザーの質問と意味的に近い文章を検索できることを確認した。

### Chunking

Markdown ドキュメントを Chunk に分割して保存。

Chunk Size や Top-K の設定によって検索結果が変化することを確認した。

### Retrieval-Augmented Generation

検索結果を Context として Gemini へ渡し、回答生成を実装。

LLM 単体では回答できないプロジェクト固有情報についても、RAG により回答可能となった。

### Prompt Engineering

回答形式を制御するプロンプトを追加。

内部情報や Markdown 構造をそのまま出力しないよう改善し、自然な日本語で回答できるようにした。

---

## Example Queries

```text
LightGBMを使ったプロジェクトを教えて
```

```text
銀行マーケティングで比較したモデルは？
```

```text
Dockerを利用したプロジェクトは？
```

```text
太陽光発電量予測で使ったモデルを教えて
```

---

## Tech Stack

* Python
* sentence-transformers
* ChromaDB
* scikit-learn
* LangChain (予定)
* Gemini API
* Streamlit (予定)

---

## Current Status

* Embedding 実装完了
* ChromaDB によるベクトル検索実装完了
* RAG パイプライン実装完了
* Gemini API 連携完了
* Prompt Engineering 実装完了
* Streamlit アプリ実装予定
* 比較実験実施予定

import os

import pandas as pd
import numpy as np
import chromadb
import streamlit as st
import yaml
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


def load_config(path_file: str) -> dict:

    with open(path_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_db(config: dict) -> chromadb.Collection:

    print("Loading DB")
    client_chromadb = chromadb.PersistentClient(path=config["vector_db"]["path"])

    return client_chromadb.get_collection(name=config["vector_db"]["collection_name"])


@st.cache_resource
def load_embedding_model(config: str) -> SentenceTransformer:
    model_name = config["embedding_model"]["model"]

    return SentenceTransformer(model_name)


def embedding_query(config: dict,
                    query: str, 
                    model_embedding: SentenceTransformer
                    ) -> np.ndarray:

    print("Embedding query")

    return model_embedding.encode(query)


def query_chromadb(collection:chromadb.Collection, 
                   query_embeddings:  np.ndarray, 
                   top_k: int) -> str:

    print("Searth query with DB")
    result = collection.query(
        query_embeddings=[query_embeddings.tolist()],
        n_results=top_k
    )

    context = ""

    for i, (doc, metadata) in enumerate(zip(result["documents"][0], result["metadatas"][0]), start=1):

        context += f"""
            【検索結果{i}】
            source: {metadata["source"]}

            {doc}

            """

    return context


def make_prompt(query: str, context: str) -> str:

    prompt = f"""
        あなたはポートフォリオ検索アシスタントです。

        以下の参考情報を基に質問へ回答してください。

        ルール:
        - 回答は自然な日本語で記述すること
        - 箇条書きは必要な場合のみ使用すること
        - 参考情報の見出しや構造をそのまま出力しないこと
        - 「プロジェクト名:」や「source:」などの内部情報は出力しないこと
        - 回答文として読みやすくまとめること
        - 情報が見つからない場合のみ「情報が見つかりませんでした」と回答すること

        参考情報:
        {context}

        質問:
        {query}
    """

    return prompt


def call_gemini_api(config: dict, prompt: str) -> str:

    load_dotenv()
    gemini_api_key = os.getenv("API_KEY")

    model_gemini = config["gemini"]["model"]
    generation_config = config["gemini"]["generation_config"]

    client_gemini = genai.Client(api_key = gemini_api_key)

    print("Call gemini api")
    response = client_gemini.models.generate_content(
        model = model_gemini, 
        config = generation_config,
        contents = prompt
    )

    # print(response.text)

    return response.text


def answer_question(query:str) -> str:

    config = load_config("./config/config.yaml")
    collection =  load_db(config=config)
    model_embedding = load_embedding_model(config=config)
    query_embedding = embedding_query(config=config, query=query, model_embedding=model_embedding)
    context = query_chromadb(collection=collection, query_embeddings=query_embedding, top_k=10)
    prompt = make_prompt(query=query, context=context)
    response = call_gemini_api(config=config, prompt=prompt)
 
    return response



######
#  rag.py単体での動作確認用
######

def main():

    query = "太陽光発電量予測で使ったモデルは？"
    answer = answer_question(query=query)
    print(f"answer is: {answer}")

if __name__ == "__main__":
    main()
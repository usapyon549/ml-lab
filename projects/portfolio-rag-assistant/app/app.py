import streamlit as st

import rag

st.title("Portfolio RAG Assistant")

query = st.text_input(
    "質問を入力してください"
)

if st.button("質問する"):
    
    answer = rag.answer_question(query=query)
    # answer = "hello world!!"
    st.write(answer)
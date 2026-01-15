import sys
import os
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import streamlit as st
from rag.retriever import answer_question


st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🏥",
    layout="centered",
)


st.markdown(
    """
    <style>
        body {
            background-color: #f8fafc;
        }
        .block-container {
            max-width: 820px;
            padding-top: 2rem;
        }
        .chat-user {
            background-color: #e0f2fe; /* soft blue */
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            color: #0f172a;
        }
        .chat-bot {
            background-color: #1e3a8a; /* dark blue */
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .sources {
    background-color: #ffffff;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    font-size: 0.9rem;
    color: #1f2937;
    margin-bottom: 1.5rem;
}
.sources ul {
    margin-top: 0.4rem;
    padding-left: 1.2rem;
}
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Healthcare Knowledge Chatbot")
st.caption(
    "Answers grounded in curated healthcare research papers using Retrieval-Augmented Generation"
)

tab_chat, tab_obs = st.tabs(["💬 Chat", "📊 Observability"])


with tab_chat:

    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.chat_input("Ask a healthcare-related question")

    if question:
        with st.spinner("Searching healthcare documents..."):
            answer, sources = answer_question(question)

        st.session_state.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )

    for item in reversed(st.session_state.history):
        st.markdown(
            f"<div class='chat-user'><strong>You</strong><br>{item['question']}</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<div class='chat-bot'><strong>Assistant</strong><br>{item['answer']}</div>",
            unsafe_allow_html=True
        )

        if item["sources"]:
            st.markdown(
                "<div class='sources'><strong>Sources</strong><ul>"
                + "".join(f"<li>{s}</li>" for s in item["sources"])
                + "</ul></div>",
                unsafe_allow_html=True
            )


with tab_obs:
    st.subheader("System Observability")

    log_path = "data/observability/logs.csv"

    if not os.path.exists(log_path):
        st.info("No interactions logged yet.")
    else:
        df = pd.read_csv(log_path)

        st.caption("Recent interactions")
        st.dataframe(df.tail(10), use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Answer vs Refusal")
            st.bar_chart(df["answer_type"].value_counts())

        with col2:
            st.caption("Latency (ms)")
            st.line_chart(df["latency_ms"])

import sys
import os
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import streamlit as st
from rag.retriever import answer_question


st.set_page_config(
    page_title="HealthWise AI",
    
    layout="centered",
)


st.markdown(
    """
    <style>
       
        .stApp {
            background: linear-gradient(180deg, #eef4ff 0%, #e6efff 100%);
            color: #1e3a5f;
        }

        header {
            background: transparent !important;
            border-bottom: none !important;
        }

        
        h1, h2, h3, h4, h5, h6, .stMarkdown {
            color: #1e3a5f;
        }

        
        .block-container {
            max-width: 820px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        
        button[data-baseweb="tab"] {
            color: #4b6b8a;
            font-weight: 500;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #3b82f6;
            border-bottom: 2px solid #3b82f6;
        }

        
        .chat-user {
            background: #f1f6ff;
            padding: 1.1rem 1.25rem;
            border-radius: 14px;
            margin-bottom: 0.6rem;
            border: 1px solid #dbeafe;
        }

        
        .chat-assistant {
            background: #f7faff;
            padding: 1.25rem;
            border-radius: 14px;
            margin-bottom: 1.4rem;

            border: 2px solid #b6c8e6;
            border-left: 5px solid #3b82f6;

            box-shadow:
                0 10px 28px rgba(59,130,246,0.14),
                0 3px 8px rgba(15,23,42,0.08);
        }

        .assistant-sources {
            font-size: 0.9rem;
            color: #4b6b8a;
        }

        
        [data-testid="stChatInput"] textarea {
            background: #f8fbff;
            border: 1.5px solid #b6c8e6;
            border-radius: 18px;
            padding: 0.9rem 1rem;
            color: #1e3a5f;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #5f7fa3;
        }

        
        [data-testid="stDataFrame"], .stChart {
            background: white;
            border-radius: 14px;
            padding: 0.75rem;
            border: 1px solid #d6e0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)





st.title("HealthWise AI")
st.caption(
    "Answers grounded in curated healthcare research papers using RAG"
)

tab_chat, tab_obs = st.tabs(["Chat", "Observability"])


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

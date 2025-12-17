import os
import streamlit as st

from rag_backend import (
    PDF_DIR,
    load_pdfs_from_dir,
    chunk_documents,
    build_or_load_vectorstore,
    build_llm,
    build_rag_chain,
    ask_with_memory,
)

st.set_page_config(page_title="NCERT Science Chatbot", layout="wide")

st.title("📘 NCERT Science Chatbot")
st.caption("Class 9 & 10 • NCERT Textbook-Based AI Assistant")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Supported Classes**
- Class 9 Science
- Class 10 Science

**Features**
- Answers strictly from NCERT textbooks
- Simple, exam-oriented explanations
- Conversational memory
""")

st.sidebar.info(
    "⏳ First-time load may take a few minutes while the NCERT books are indexed. "
    "Subsequent loads are instant."
)

# -------------------------------------------------
# Session State
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

from rag_backend import FAISS_DIR

if "rag_chain" not in st.session_state:

    if FAISS_DIR.exists():
        # Fast path: FAISS already built
        docs = None  # not needed
        chunks = None
        vectorstore = build_or_load_vectorstore([])
        llm = build_llm()
        st.session_state.rag_chain = build_rag_chain(vectorstore, llm)

    else:
        # Slow path: first-time build
        with st.spinner("Setting up NCERT knowledge base (one-time process)..."):
            docs = load_pdfs_from_dir(PDF_DIR)
            chunks = chunk_documents(docs)
            vectorstore = build_or_load_vectorstore(chunks)
            llm = build_llm()
            st.session_state.rag_chain = build_rag_chain(vectorstore, llm)


# -------------------------------------------------
# Chat History
# -------------------------------------------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["bot"])

# -------------------------------------------------
# Chat Input
# -------------------------------------------------
question = st.chat_input("Ask a question from NCERT Science...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        answer = ask_with_memory(
            st.session_state.rag_chain,
            question,
            st.session_state.history,
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.caption("Answers are generated strictly from NCERT textbook content.")

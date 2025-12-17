import os
import streamlit as st

from rag_backend import (
    PDF_DIR,
    load_pdfs_from_dir,
    chunk_documents,
    build_vectorstore,
    build_llm,
    build_rag_chain,
    ask_with_memory,
)

st.set_page_config(page_title="NCERT Science Chatbot", layout="wide")

st.title("📘 NCERT Science Chatbot")
st.caption("Class 9 & 10 • Textbook-based AI Assistant")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**Supported Classes**
- Class 9 Science
- Class 10 Science

**Purpose**
- NCERT-based exam preparation
- Concept clarity using textbook content
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Example Questions**")
st.sidebar.markdown("""
- What is photosynthesis?
- Explain laws of motion
- What are acids, bases and salts?
- What is reflection of light?
""")

# -----------------------------
# Session State
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "rag_chain" not in st.session_state:

    if "OPENAI_API_KEY" not in os.environ:
        st.error("OPENAI_API_KEY not set")
        st.stop()

    with st.spinner("Loading NCERT textbooks..."):
        docs = load_pdfs_from_dir(PDF_DIR)
        chunks = chunk_documents(docs)
        vectorstore = build_vectorstore(chunks)
        llm = build_llm()
        st.session_state.rag_chain = build_rag_chain(vectorstore, llm)

# -----------------------------
# Chat History
# -----------------------------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["bot"])

# -----------------------------
# Chat Input
# -----------------------------
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

    st.caption("Answer generated using NCERT textbook content only.")

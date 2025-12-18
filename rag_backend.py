from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# -------------------------------------------------
# Paths & Models
# -------------------------------------------------
BASE_DIR = Path("data")
PDF_DIR = BASE_DIR / "pdfs"
FAISS_DIR = BASE_DIR / "faiss_index"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# -------------------------------------------------
# Load PDFs (LOCAL ONLY)
# -------------------------------------------------
def load_pdfs_from_dir(pdf_dir: Path) -> List[Document]:
    docs = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError("No PDFs found in data/pdfs/")

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            text = page.page_content.strip()
            if len(text) < 200:
                continue

            page.metadata["source"] = pdf.name
            docs.append(page)

    print(f"Loaded {len(docs)} meaningful pages")
    return docs


# -------------------------------------------------
# Chunking
# -------------------------------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


# -------------------------------------------------
# FAISS Vector Store (BATCHED + CACHED)
# -------------------------------------------------
def build_or_load_vectorstore(chunks: List[Document]) -> FAISS:
    

    # 🔹 Load cached index if exists
    if FAISS_DIR.exists():
        print("Loading FAISS index from disk...")
        return FAISS.load_local(
            FAISS_DIR,
            OpenAIEmbeddings(model=EMBEDDING_MODEL),
            allow_dangerous_deserialization=True,
        )

    print("Building FAISS index with batching (first run)...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    BATCH_SIZE = 40  # safe for OpenAI limits

    # 🔹 Initialize FAISS with FIRST batch (CRITICAL)
    first_batch = chunks[:BATCH_SIZE]
    vectorstore = FAISS.from_documents(first_batch, embeddings)

    print(f"Embedded {len(first_batch)}/{len(chunks)}")

    # 🔹 Add remaining batches
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    # 🔹 Save index
    vectorstore.save_local(FAISS_DIR)
    print("FAISS index saved to disk")

    return vectorstore



# -------------------------------------------------
# LLM + RAG Chain
# -------------------------------------------------
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=500,
    )


def build_rag_chain(vectorstore: FAISS, llm: ChatOpenAI):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 9},
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('source')} | Page {d.metadata.get('page', 'N/A')}]\n"
            f"{d.page_content}"
            for d in docs
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an NCERT Science assistant for Indian school students "
                "(Class 9 and Class 10).\n\n"
                "Rules:\n"
                "- Answer ONLY from the NCERT textbook content.\n"
                "- User can ask questions in a vague manner; \n"
                "- User may ask question in Hindi or English; respond in the same language.\n"
                "- Explain in simple, exam-oriented language.\n"
                "- If the answer is not found, say:\n"
                "  'This topic is not covered in the NCERT textbook.'\n\n"
                "NCERT Content:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# -------------------------------------------------
# Conversational Memory
# -------------------------------------------------
def ask_with_memory(
    rag_chain,
    question: str,
    history: List[Dict[str, str]],
    max_turns: int = 4,
) -> str:

    if history:
        convo = "\n".join(
            f"User: {h['user']}\nAssistant: {h['bot']}"
            for h in history[-max_turns:]
        )
        question = f"{convo}\n\nQuestion: {question}"

    answer = rag_chain.invoke(question)

    history.append({"user": question, "bot": answer})
    history[:] = history[-max_turns:]

    return answer

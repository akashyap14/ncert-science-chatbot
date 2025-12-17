from pathlib import Path
from typing import List, Dict
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path("data")
PDF_DIR = BASE_DIR / "pdfs"
VECTORSTORE_PATH = Path("data/vectorstore")
VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# -----------------------------
# Load PDFs from local path
# -----------------------------
def load_pdfs_from_dir(pdf_dir: Path) -> List[Document]:
    docs = []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(
            "No PDFs found in data/pdfs/. "
            "Please add NCERT Class 9 & 10 Science PDFs."
        )

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            text = page.page_content.strip()

            # Skip empty / index pages
            if len(text) < 200:
                continue

            # Add source info
            page.metadata["source"] = pdf.name
            docs.append(page)

    print(f"Loaded {len(docs)} meaningful pages")
    return docs


# -----------------------------
# Chunking
# -----------------------------
def chunk_documents(
    docs: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


# -----------------------------
# Vector Store (SAFE BATCHING)
# -----------------------------
# def build_vectorstore(chunks: List[Document]) -> InMemoryVectorStore:
#     embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
#     vectorstore = InMemoryVectorStore(embedding=embeddings)

#     BATCH_SIZE = 50

#     for i in range(0, len(chunks), BATCH_SIZE):
#         batch = chunks[i : i + BATCH_SIZE]
#         vectorstore.add_documents(batch)
#         print(f"Embedded {i + len(batch)} / {len(chunks)}")

#     return vectorstore


def build_or_load_vectorstore(chunks: List[Document]) -> InMemoryVectorStore:
    cache_file = VECTORSTORE_PATH / "vectorstore.pkl"

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 🔹 Load if exists
    if cache_file.exists():
        print("Loading cached vectorstore from disk...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # 🔹 Build once
    print("Building vectorstore (first time only)...")
    vectorstore = InMemoryVectorStore(embedding=embeddings)

    BATCH_SIZE = 50
    for i in range(0, len(chunks), BATCH_SIZE):
        vectorstore.add_documents(chunks[i:i+BATCH_SIZE])
        print(f"Embedded {i + BATCH_SIZE} / {len(chunks)}")

    # 🔹 Save to disk
    with open(cache_file, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Vectorstore cached successfully.")
    return vectorstore


# -----------------------------
# LLM + RAG Chain
# -----------------------------
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=500,
    )


def build_rag_chain(vectorstore: InMemoryVectorStore, llm: ChatOpenAI):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"[{doc.metadata.get('source')} | Page {doc.metadata.get('page', 'N/A')}]\n"
            f"{doc.page_content}"
            for doc in docs
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an NCERT Science assistant for Indian school students "
                "(Class 9 and Class 10).\n\n"
                "Rules:\n"
                "- Answer ONLY from the provided NCERT textbook content.\n"
                "- Explain concepts in simple, exam-oriented language.\n"
                "- If the answer is not found in NCERT, say clearly:\n"
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


# -----------------------------
# Conversational Memory
# -----------------------------
def ask_with_memory(
    rag_chain,
    question: str,
    history: List[Dict[str, str]],
    max_turns: int = 4,
) -> str:

    if history:
        recent = history[-max_turns:]
        convo = "\n".join(
            f"User: {t['user']}\nAssistant: {t['bot']}"
            for t in recent
        )
        question = f"Previous conversation:\n{convo}\n\nQuestion:\n{question}"

    answer = rag_chain.invoke(question)

    history.append({"user": question, "bot": answer})
    history[:] = history[-max_turns:]

    return answer

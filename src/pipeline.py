import os
import tempfile

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from src.config import (
    CHUNK_BUFFER_SIZE,
    CHUNK_MIN_SIZE,
    CHUNK_THRESHOLD_AMOUNT,
    CHUNK_THRESHOLD_TYPE,
)


def process_pdf(uploaded_file, embeddings):
    """Tạo retriever từ file PDF và trả về số lượng chunk."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getvalue())
        path = f.name

    docs = PyPDFLoader(path).load()
    chunks = SemanticChunker(
        embeddings=embeddings,
        buffer_size=CHUNK_BUFFER_SIZE,
        breakpoint_threshold_type=CHUNK_THRESHOLD_TYPE,
        breakpoint_threshold_amount=CHUNK_THRESHOLD_AMOUNT,
        min_chunk_size=CHUNK_MIN_SIZE,
        add_start_index=True,
    ).split_documents(docs)

    retriever = Chroma.from_documents(
        documents=chunks, embedding=embeddings
    ).as_retriever()

    os.unlink(path)
    return retriever, len(chunks)


def retrieve_docs(question, retriever):
    """Truy xuất tài liệu liên quan từ vector DB."""
    docs = retriever.invoke(question)
    return "\n\n".join(d.page_content for d in docs)


def parse_answer(output):
    """Trích xuất câu trả lời từ output của LLM."""
    if "Answer:" in output:
        return output.split("Answer:")[1].strip()
    return output.strip()

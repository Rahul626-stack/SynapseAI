"""RAG Pipeline — Document ingestion, embedding, storage, and retrieval.

Uses PyMuPDF for PDF extraction, LangChain for semantic chunking,
SentenceTransformers for local embeddings, and ChromaDB for persistent
vector storage.
"""

import hashlib
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Constants ────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_PATH = ".chroma_store"
DISTANCE_THRESHOLD = 1.3


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def _get_embed_model():
    """Returns cached SentenceTransformer instance (loads once per session)."""
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def _get_client():
    """Returns cached ChromaDB PersistentClient (persists across reruns)."""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _collection_id(original_filename: str) -> str:
    """Generate deterministic collection name from original filename.

    Uses MD5 hash for deduplication — same file always maps to the same
    collection, enabling instant re-quizzing without re-ingestion.
    """
    return "quiz_" + hashlib.md5(original_filename.encode()).hexdigest()[:10]


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str, original_filename: str = "document.pdf") -> str:
    """Full RAG ingestion pipeline: extract → chunk → embed → store.

    Args:
        pdf_path: Absolute path to the PDF file on disk.
        original_filename: Original upload filename (used for deduplication).

    Returns:
        Collection name (str).
    """
    client = _get_client()
    col_name = _collection_id(original_filename)

    # ── Deduplication: skip if collection already has data ────────────────
    try:
        existing = client.get_collection(name=col_name)
        if existing.count() > 0:
            return col_name
    except Exception:
        pass

    # ── Extract text with PyMuPDF (page by page) ─────────────────────────
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
    except Exception as e:
        full_text = f"[Error reading PDF: {e}]"

    # ── Chunk with LangChain RecursiveCharacterTextSplitter ──────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        chunks = [full_text]

    # ── Embed with SentenceTransformers (local, CPU) ─────────────────────
    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # ── Store in ChromaDB ────────────────────────────────────────────────
    # Delete existing collection to allow clean re-ingestion
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.create_collection(name=col_name)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )

    return col_name


# ── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_context(col_name: str, query: str, top_k: int = 5) -> str:
    """Retrieve top-k relevant chunks for a query from ChromaDB.

    Uses L2 distance filtering with DISTANCE_THRESHOLD to exclude
    semantically irrelevant chunks.

    Special "Entire Document" mode: when query is empty or matches
    "entire document", fetches top-20 chunks directly without embedding.

    Args:
        col_name: ChromaDB collection name.
        query: Search query / topic string.
        top_k: Number of results to retrieve.

    Returns:
        Joined context string of relevant chunks.
    """
    client = _get_client()
    collection = client.get_collection(name=col_name)

    # ── "Entire Document" mode — no query embedding needed ───────────────
    if not query or query.strip().lower() in ("", "entire document"):
        results = collection.get(limit=20)
        chunks = results["documents"] if results["documents"] else []
        return "\n\n---\n\n".join(chunks)

    # ── Generate query embedding ─────────────────────────────────────────
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(query, show_progress_bar=False).tolist()

    # ── Perform L2 distance search ───────────────────────────────────────
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    # ── Filter by distance threshold ─────────────────────────────────────
    context_chunks = []
    documents = results["documents"][0]
    distances = results["distances"][0] if results.get("distances") else [0] * len(documents)

    for doc, dist in zip(documents, distances):
        if dist <= DISTANCE_THRESHOLD:
            context_chunks.append(doc)

    # ── Fallback: return closest chunk if all exceed threshold ────────────
    if not context_chunks and documents:
        context_chunks = [documents[0]]

    return "\n\n---\n\n".join(context_chunks)

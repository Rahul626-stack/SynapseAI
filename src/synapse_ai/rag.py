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
    """Returns cached SentenceTransformer instance (loaded once, cached for session)."""
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

    Uses SHA256 hash for deduplication — same file always maps to the same
    collection, enabling instant re-quizzing without re-ingestion.
    """
    return "quiz_" + hashlib.sha256(original_filename.encode()).hexdigest()[:12]


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

    # ── Extract text with PyMuPDF (block-level for structure preservation) ──
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0:  # text block
                    lines_text = ""
                    for line in block["lines"]:
                        span_text = " ".join(span["text"] for span in line["spans"])
                        lines_text += span_text + " "
                    full_text += lines_text.strip() + "\n\n"
        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

    # Guard against empty/scanned PDFs
    if not full_text.strip():
        raise ValueError("PDF appears to be empty or contains only images/scanned content.")

    # ── Chunk with LangChain RecursiveCharacterTextSplitter ──────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        chunks = [full_text]

    # ── Embed with SentenceTransformers (local, CPU) ─────────────────────
    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=64).tolist()

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
    semantically irrelevant chunks. Introduces random sampling from a 
    larger pool of chunks to ensure high variability between quiz generations.

    Special "Entire Document" mode: when query is empty or matches
    "entire document", fetches up to 30 chunks and samples from them.

    Args:
        col_name: ChromaDB collection name.
        query: Search query / topic string.
        top_k: Number of results to retrieve.

    Returns:
        Joined context string of relevant chunks.
    """
    import random
    client = _get_client()
    collection = client.get_collection(name=col_name)

    # ── "Entire Document" mode — no query embedding needed ───────────────
    if not query or query.strip().lower() in ("", "entire document"):
        results = collection.get(limit=30)
        chunks = results["documents"] if results["documents"] else []
        if len(chunks) > top_k:
            chunks = random.sample(chunks, top_k)
        return "\n\n---\n\n".join(chunks)

    # ── Generate query embedding ─────────────────────────────────────────
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(query, show_progress_bar=False).tolist()

    # ── Perform L2 distance search (fetch larger pool for variety) ───────
    pool_size = max(top_k * 2, 15)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=pool_size,
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

    # ── Randomly sample top_k from the valid pool ────────────────────────
    if len(context_chunks) > top_k:
        context_chunks = random.sample(context_chunks, top_k)

    return "\n\n---\n\n".join(context_chunks)

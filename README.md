# 🧠 synapse.ai (synapse.ai)

**Intelligent, AI-Powered Quiz Generation Platform**

Transform any PDF document into a pedagogically-sound, multi-format assessment using RAG, multi-agent AI, and Bloom's Taxonomy.

---

## ✨ Features

- **RAG-Powered** — Questions are grounded in your actual document content via semantic retrieval (ChromaDB + SentenceTransformers)
- **Bloom's Taxonomy** — Questions distributed across 6 cognitive levels (Remember → Create) for balanced assessments
- **Multi-Agent AI** — Two specialized CrewAI agents collaborate: Content Analyzer → Quiz Generator
- **Multiple Question Types** — MCQ, True/False, and Short Answer
- **Premium UI** — Glassmorphic dark-themed Streamlit dashboard with Plotly analytics
- **Cognitive Analytics** — Radar charts, bar charts, and per-level accuracy breakdowns

---

## 🏗️ Architecture

```
PDF Upload → PyMuPDF Extraction → LangChain Chunking → SentenceTransformer Embedding → ChromaDB Storage
                                                                                          ↓
User Topic → Semantic Retrieval (top-k, distance-filtered) → CrewAI Pipeline → Pydantic Output
                                                                                          ↓
                                                              Streamlit UI → Quiz Form → Score Analytics
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq (Llama-3.3 70B Versatile) |
| **Agent Framework** | CrewAI (≥0.121.1) |
| **Embedding Model** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | ChromaDB (PersistentClient) |
| **PDF Parsing** | PyMuPDF (`fitz`) |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Frontend** | Streamlit + Custom CSS (Glassmorphism) |
| **Charting** | Plotly (Radar + Bar charts) |
| **Validation** | Pydantic |

---

## 📁 Project Structure

```
synapse.ai/
├── .env                          # API keys (GROQ_API_KEY)
├── .chroma_store/                # Persistent ChromaDB data
├── pyproject.toml                # Project config & dependencies
├── requirements.txt              # Pip requirements
├── test.pdf                      # Sample test document
├── knowledge/                    # CrewAI knowledge base
├── src/
│   └── synapse_ai/
│       ├── __init__.py
│       ├── app.py                # Streamlit frontend (main)
│       ├── rag.py                # RAG pipeline
│       ├── crew.py               # CrewAI agents & tasks
│       ├── blooms.py             # Bloom's Taxonomy engine
│       ├── main.py               # CLI entry point
│       ├── config/
│       │   ├── agent.yaml        # Agent definitions
│       │   └── task.yaml         # Task definitions
│       ├── models/
│       │   └── quiz.py           # Pydantic schemas
│       └── tools/
│           └── extract_pdf_content_tool.py
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install uv
crewai install
```

Or using pip directly:

```bash
pip install -r requirements.txt
```

### 2. Set API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Launch the app

```bash
streamlit run src/synapse_ai/app.py
```

---

## 📊 Bloom's Taxonomy Distribution

Default distribution for a 10-question quiz:

| Level | Count | Cognitive Verbs |
|-------|-------|----------------|
| Remember | 2 | recall, list, define, identify |
| Understand | 3 | explain, summarize, paraphrase |
| Apply | 2 | use, solve, demonstrate |
| Analyze | 2 | compare, differentiate, examine |
| Evaluate | 1 | judge, justify, critique |
| Create | 0 | design, construct, formulate |

Distribution scales proportionally for any question count (3–20).

---

## ⚡ Key Parameters

| Parameter | Value |
|-----------|-------|
| Embedding dimensions | 384 |
| Chunk size / overlap | 1000 / 200 |
| L2 distance threshold | 1.3 |
| LLM temperature | 0.7 |
| Default questions | 10 (configurable 3–20) |

---

## 👤 Author

**Rahul** — AI Engineer

---

## 📄 License

This project is for educational purposes.

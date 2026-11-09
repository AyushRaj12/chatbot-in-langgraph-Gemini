# LangGraph Chatbot

A public Streamlit chatbot built with LangGraph and Google Gemini.

## Features

- Live web and recent-news search with source URLs
- Structured stock quotes through Alpha Vantage, with web-search fallback
- Calculator tool
- PDF retrieval-augmented generation (RAG) using Gemini embeddings and FAISS
- Bounded, in-memory SQLite LangGraph conversation checkpoints
- Streaming responses and visible tool activity
- Local Model Context Protocol (MCP) tools for safe, deterministic utilities

## Run locally

Use Python 3.12 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Replace `GOOGLE_API_KEY` in `.env` with a Gemini API key, then run:

```bash
streamlit run streamlit_frontend.py --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501`.

## Streamlit Community Cloud

Deploy `streamlit_frontend.py` and configure these root-level secrets:

```toml
GOOGLE_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
```

Optional structured stock quotes:

```toml
ALPHA_VANTAGE_API_KEY = "your-alpha-vantage-key"
```

Without an Alpha Vantage key, the assistant uses web search for public market information and identifies that results may be delayed.

## MCP behavior

`mcp_tools_server.py` runs locally over standard input/output and exposes only deterministic, read-only tools:

- Temperature conversion
- Percentage calculation
- Current UTC time

The MCP subprocess receives no Gemini, stock, or other application credentials. The public app does not connect to remote MCP servers or expose private and state-changing tools.

## Public deployment limits and privacy

- Anyone with the public URL can consume the Gemini and search quotas.
- Best-effort per-process limits are 200 chat turns, 400 searches, 100 stock lookups, and 40 PDF indexes per UTC day. Cold starts and additional replicas reset these counters, so keep the Gemini project on its free tier or configure provider-side hard quotas before enabling billing.
- Each browser session is limited to 30 attempted chat turns, including failed calls.
- Streamlit rejects uploads over 5 MB before the application reads them.
- PDF extraction runs in an isolated subprocess with memory, CPU, output, page, and time limits.
- PDFs are limited to 30 pages, 100,000 extracted characters, and 120 chunks.
- Uploaded PDF text is sent to Gemini for embeddings and responses. Visitors should not upload confidential or sensitive documents.
- Conversation checkpoints use bounded in-memory SQLite storage. Inactive threads expire after one hour and no more than 200 threads are retained.
- PDF indexes are held in process memory and follow the same thread retention limits.
- Free-tier Gemini prompts and responses may be used to improve Google products.
- Keep `.env` and `.streamlit/secrets.toml` out of Git.

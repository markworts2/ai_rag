---
name: ollama-rag-tuner
description: >
  Tune a RAG (Retrieval-Augmented Generation) pipeline using Ollama running on
  a Raspberry Pi over the local network and ChromaDB as the vector store.
  Use this skill when the user wants to ingest documents into ChromaDB, configure
  embeddings via Ollama, run RAG queries, or tune chunk/retrieval settings.
---

# Ollama RAG Tuner

A skill for building and tuning a RAG pipeline that connects to:
- **Ollama** running on a Raspberry Pi (local network IP)
- **ChromaDB** as the vector store (local or also on the Pi)

---

## Overview

The skill handles the full RAG lifecycle:

1. **Connect** – verify Ollama and ChromaDB are reachable
2. **Ingest** – chunk documents and embed them into ChromaDB via Ollama embeddings
3. **Query** – retrieve relevant chunks and generate answers with Ollama
4. **Tune** – adjust chunking, overlap, top-k, and prompt templates
5. **Evaluate** – score answer quality against expected outputs

---

## Environment Setup

Before writing any code, install the required Python packages:

```bash
pip install chromadb ollama httpx --break-system-packages
```

Or detect and install only what's missing:

```python
import importlib, subprocess, sys

REQUIRED = ["chromadb", "ollama", "httpx"]
for pkg in REQUIRED:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages"])
```

---

## Configuration

Always prompt the user for (or infer from context):

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://192.168.1.169:11434` | Raspberry Pi Ollama endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.2` | Ollama chat/generation model |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8000` | ChromaDB port (use `None` for in-process) |
| `COLLECTION` | `rag_docs` | ChromaDB collection name |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |

Expose these as constructor args or environment variables — never hardcode them.

---

## Architecture Pattern

```
User Query
    │
    ▼
[Ollama Embed]  ──►  [ChromaDB Query]  ──►  Top-K Chunks
                                                  │
                                                  ▼
                                         [Ollama Chat + Context]
                                                  │
                                                  ▼
                                             Answer
```

---

## Core Implementation

### 1. Connection Check

Always verify connectivity before proceeding:

```python
import httpx

def check_ollama(host: str) -> bool:
    try:
        r = httpx.get(f"{host}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Cannot reach Ollama at {host}: {e}")
        return False

def check_chroma(client) -> bool:
    try:
        client.heartbeat()
        return True
    except Exception as e:
        print(f"❌ ChromaDB not reachable: {e}")
        return False
```

If either check fails, print a clear diagnostic:
- For Ollama: remind the user to confirm `ollama serve` is running on the Pi and that port 11434 is open (`sudo ufw allow 11434` on the Pi if needed).
- For ChromaDB: remind the user to run `chroma run --host 0.0.0.0 --port 8000` if using HTTP mode.

### 2. Embedding via Ollama

```python
import ollama

def embed(text: str, model: str, host: str) -> list[float]:
    client = ollama.Client(host=host)
    response = client.embeddings(model=model, prompt=text)
    return response["embedding"]
```

Batch embed when ingesting multiple chunks — don't call one-by-one in a tight loop without rate control.

### 3. Chunking Strategy

Use a simple character splitter by default. Offer sentence-aware splitting as an upgrade:

```python
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]
```

For better quality, prefer splitting on `\n\n` (paragraph boundaries) first, then fall back to character splitting.

### 4. ChromaDB Ingestion

```python
import chromadb
import hashlib

def ingest(texts: list[str], source: str, collection, embed_fn) -> int:
    """Embed and store chunks. Returns number of chunks added."""
    chunks = []
    for text in texts:
        chunks.extend(chunk_text(text))
    
    ids = [hashlib.md5(c.encode()).hexdigest() for c in chunks]
    embeddings = [embed_fn(c) for c in chunks]
    metadatas = [{"source": source, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)
```

Use `upsert` not `add` — it's idempotent, so re-ingesting the same document won't create duplicates.

### 5. RAG Query

```python
def rag_query(query: str, collection, embed_fn, chat_model: str, ollama_host: str,
              top_k: int = 5, system_prompt: str = None) -> str:
    
    # Retrieve
    q_embedding = embed_fn(query)
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    docs = results["documents"][0]
    
    # Build context
    context = "\n\n---\n\n".join(docs)
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY "
            "the provided context. If the context doesn't contain the answer, say "
            "'I don't have enough information to answer that.'"
        )
    
    user_message = f"Context:\n{context}\n\nQuestion: {query}"
    
    # Generate
    client = ollama.Client(host=ollama_host)
    response = client.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    return response["message"]["content"]
```

---

## Tuning Parameters

When the user asks to "tune" or "improve" the RAG pipeline, systematically vary these:

### Chunking
- Increase `chunk_size` (512 → 1024 → 2048) for dense technical docs
- Decrease for FAQ-style content
- Try paragraph-aware splitting over character splitting

### Retrieval
- Increase `top_k` (5 → 10 → 20) if answers are incomplete
- Add **metadata filtering** (e.g., filter by `source`) to narrow context
- Consider **MMR (Maximal Marginal Relevance)** via ChromaDB to reduce redundant chunks

### Prompt Engineering
- Add `few_shot` examples to the system prompt
- Instruct the model to cite sources: `"Always mention the source document name."`
- Add a chain-of-thought instruction for complex questions

### Model Selection
- Try `nomic-embed-text` vs `mxbai-embed-large` for embeddings
- Try `llama3.2` vs `mistral` vs `phi3` for generation
- Smaller models (phi3, gemma2:2b) are faster on Pi but less accurate

---

## Complete Runnable Script Template

When asked to create a working RAG tool, generate a single self-contained Python script following this structure:

```
1. Imports + dependency check
2. Config dataclass or argparse
3. check_ollama() + check_chroma()  →  exit if either fails
4. ChromaDB client + collection setup
5. embed() helper (wraps Ollama)
6. chunk_text() helper
7. ingest(files_or_text) function
8. rag_query(question) function
9. CLI entrypoint:
   - `python rag.py ingest path/to/docs/`
   - `python rag.py query "What is X?"`
   - `python rag.py tune --chunk-size 1024 --top-k 10`
```

---

## ChromaDB Modes

| Mode | When to use | How to connect |
|---|---|---|
| **In-process** (embedded) | Local dev, no server needed | `chromadb.Client()` |
| **HTTP server** | Remote or persistent | `chromadb.HttpClient(host=..., port=...)` |
| **Persistent local** | Local + persistent without server | `chromadb.PersistentClient(path="./chroma_db")` |

Prefer **PersistentClient** for Raspberry Pi deployments — it avoids running a separate server process.

If ChromaDB is also on the Pi, use `HttpClient` and ensure port 8000 is accessible.

---

## Error Handling

Always handle these failure modes gracefully:

| Error | Likely Cause | Action |
|---|---|---|
| `httpx.ConnectError` | Pi not reachable | Check IP, check `ollama serve` |
| `ollama.ResponseError` | Model not pulled | Run `ollama pull <model>` on the Pi |
| `chromadb.errors.InvalidCollectionException` | Wrong collection name | Create or list collections |
| Embedding dimension mismatch | Changed embed model | Delete and recreate collection |
| Slow responses (>10s) | Pi CPU overloaded | Use smaller model, reduce chunk count |

---

## Raspberry Pi Tips

- **Find the Pi's IP**: `ping raspberrypi.local` or check your router's DHCP table.
- **Keep Ollama running after SSH disconnect**: Use `tmux` or create a systemd service.
- **Pull models on the Pi first**: `ollama pull nomic-embed-text && ollama pull llama3.2`
- **Check available models**: `curl http://<PI_IP>:11434/api/tags`
- **Monitor Pi resources**: `htop` on the Pi — phi3 and gemma2:2b are gentler on RAM.

---

## Evals

To evaluate RAG quality, build a small question-answer test set:

```json
[
  {
    "question": "What is X?",
    "expected_keywords": ["keyword1", "keyword2"],
    "source_doc": "doc_name.txt"
  }
]
```

Score by:
1. **Keyword recall** – does the answer contain expected keywords?
2. **Source grounding** – was the correct source doc retrieved?
3. **Latency** – end-to-end time including Pi round trip

```python
def score_answer(answer: str, expected_keywords: list[str]) -> float:
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    return hits / len(expected_keywords)
```
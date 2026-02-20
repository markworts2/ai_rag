#!/usr/bin/env python3
"""
rag.py — Ollama + ChromaDB RAG pipeline for Raspberry Pi
=========================================================
Usage:
    python rag.py ingest ./docs/           # Ingest all .txt/.md/.pdf files
    python rag.py query "What is X?"       # Ask a question
    python rag.py tune                     # Interactive tuning session
    python rag.py status                   # Check connections & collection stats
    python rag.py eval ./evals.json        # Run a question-answer eval set

Configuration (env vars or edit DEFAULTS below):
    OLLAMA_HOST     e.g. http://192.168.1.100:11434
    EMBED_MODEL     e.g. nomic-embed-text
    CHAT_MODEL      e.g. llama3.2
    CHROMA_PATH     local path for persistent ChromaDB  (default: ./chroma_db)
    COLLECTION      ChromaDB collection name            (default: rag_docs)
    CHUNK_SIZE      characters per chunk                (default: 512)
    CHUNK_OVERLAP   overlap between chunks              (default: 64)
    TOP_K           chunks to retrieve per query        (default: 5)
"""

import os
import sys
import json
import time
import hashlib
import argparse
import importlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# ─── Dependency bootstrap ───────────────────────────────────────────────────

REQUIRED_PACKAGES = {"chromadb": "chromadb", "ollama": "ollama", "httpx": "httpx"}

def _ensure_deps():
    missing = []
    for mod, pkg in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(pkg)
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "--break-system-packages"],
            stdout=subprocess.DEVNULL,
        )
        print("✅ Packages installed.\n")

_ensure_deps()

import httpx
import chromadb
import ollama as ollama_lib

# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    ollama_host:   str = os.getenv("OLLAMA_HOST",   "http://raspberrypi.local:11434")
    embed_model:   str = os.getenv("EMBED_MODEL",   "nomic-embed-text")
    chat_model:    str = os.getenv("CHAT_MODEL",    "llama3.2")
    chroma_path:   str = os.getenv("CHROMA_PATH",   "./chroma_db")
    collection:    str = os.getenv("COLLECTION",    "rag_docs")
    chunk_size:    int = int(os.getenv("CHUNK_SIZE",    "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k:         int = int(os.getenv("TOP_K",         "5"))

CFG = Config()

# ─── Connectivity ────────────────────────────────────────────────────────────

def check_ollama(cfg: Config) -> bool:
    try:
        r = httpx.get(f"{cfg.ollama_host}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"✅ Ollama reachable at {cfg.ollama_host}")
            print(f"   Available models: {', '.join(models) or '(none pulled yet)'}")
            return True
    except Exception as e:
        print(f"❌ Cannot reach Ollama at {cfg.ollama_host}")
        print(f"   Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. SSH into your Pi and run: ollama serve")
        print(f"   2. Make sure port 11434 is open: sudo ufw allow 11434")
        print(f"   3. Find your Pi's IP: ping raspberrypi.local")
        print(f"   4. Set OLLAMA_HOST=http://<PI_IP>:11434")
    return False

def get_chroma(cfg: Config) -> chromadb.PersistentClient:
    client = chromadb.PersistentClient(path=cfg.chroma_path)
    print(f"✅ ChromaDB ready at {cfg.chroma_path}")
    return client

def get_collection(client: chromadb.PersistentClient, cfg: Config):
    col = client.get_or_create_collection(
        name=cfg.collection,
        metadata={"hnsw:space": "cosine"},
    )
    return col

# ─── Embedding ───────────────────────────────────────────────────────────────

def embed(text: str, cfg: Config) -> list[float]:
    client = ollama_lib.Client(host=cfg.ollama_host)
    try:
        response = client.embeddings(model=cfg.embed_model, prompt=text)
        return response["embedding"]
    except ollama_lib.ResponseError as e:
        if "not found" in str(e).lower():
            print(f"\n❌ Embedding model '{cfg.embed_model}' not found on the Pi.")
            print(f"   Run on Pi: ollama pull {cfg.embed_model}")
            sys.exit(1)
        raise

# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking with character fallback."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                # Hard split long paragraphs
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + chunk_size])
                    start += chunk_size - overlap
                current = ""

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c.strip()) > 20]

# ─── File loading ─────────────────────────────────────────────────────────────

def load_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".rst", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print(f"   ⚠️  Skipping {path.name} — install pypdf: pip install pypdf --break-system-packages")
    else:
        print(f"   ⚠️  Skipping unsupported file type: {path.name}")
    return None

# ─── Ingest ───────────────────────────────────────────────────────────────────

def cmd_ingest(args, cfg: Config):
    if not check_ollama(cfg):
        sys.exit(1)

    chroma = get_chroma(cfg)
    col = get_collection(chroma, cfg)

    source_path = Path(args.source)
    files = list(source_path.rglob("*")) if source_path.is_dir() else [source_path]
    files = [f for f in files if f.is_file()]

    total_chunks = 0
    for fpath in files:
        text = load_file(fpath)
        if text is None:
            continue

        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        print(f"   📄 {fpath.name}: {len(chunks)} chunks", end="", flush=True)

        ids, embeddings, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            uid = hashlib.md5(f"{fpath.name}::{i}::{chunk[:50]}".encode()).hexdigest()
            emb = embed(chunk, cfg)
            ids.append(uid)
            embeddings.append(emb)
            documents.append(chunk)
            metadatas.append({"source": fpath.name, "chunk_index": i})
            print(".", end="", flush=True)

        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f" ✓")

    print(f"\n✅ Ingested {total_chunks} total chunks into collection '{cfg.collection}'")
    print(f"   Collection now has {col.count()} total documents.")

# ─── Query ───────────────────────────────────────────────────────────────────

def cmd_query(args, cfg: Config):
    if not check_ollama(cfg):
        sys.exit(1)

    chroma = get_chroma(cfg)
    col = get_collection(chroma, cfg)

    if col.count() == 0:
        print("❌ Collection is empty. Run `python rag.py ingest <path>` first.")
        sys.exit(1)

    question = args.question
    print(f"\n🔍 Query: {question}")
    print(f"   Retrieving top {cfg.top_k} chunks...\n")

    t0 = time.time()
    q_emb = embed(question, cfg)
    results = col.query(query_embeddings=[q_emb], n_results=min(cfg.top_k, col.count()))
    docs = results["documents"][0]
    sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
    t_retrieve = time.time() - t0

    if args.show_chunks:
        for i, (doc, src) in enumerate(zip(docs, sources)):
            print(f"  [{i+1}] ({src})\n  {doc[:200]}{'...' if len(doc) > 200 else ''}\n")

    context = "\n\n---\n\n".join(docs)
    system_prompt = (
        "You are a precise, helpful assistant. Answer the user's question using ONLY "
        "the provided context. If the context doesn't contain enough information, say "
        "'I don't have enough information in my knowledge base to answer that.' "
        "Always be concise and accurate."
    )
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    print("💬 Generating answer...\n")
    t1 = time.time()
    client = ollama_lib.Client(host=cfg.ollama_host)
    try:
        response = client.chat(
            model=cfg.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        answer_parts = []
        for chunk in response:
            part = chunk["message"]["content"]
            print(part, end="", flush=True)
            answer_parts.append(part)
        print()
        answer = "".join(answer_parts)
    except ollama_lib.ResponseError as e:
        if "not found" in str(e).lower():
            print(f"\n❌ Chat model '{cfg.chat_model}' not found on the Pi.")
            print(f"   Run on Pi: ollama pull {cfg.chat_model}")
            sys.exit(1)
        raise

    t_generate = time.time() - t1
    unique_sources = list(dict.fromkeys(sources))

    print(f"\n📎 Sources: {', '.join(unique_sources)}")
    print(f"⏱  Retrieve: {t_retrieve:.2f}s | Generate: {t_generate:.2f}s | Total: {t_retrieve + t_generate:.2f}s")

# ─── Status ───────────────────────────────────────────────────────────────────

def cmd_status(args, cfg: Config):
    print("=" * 50)
    print("  RAG Pipeline Status")
    print("=" * 50)
    print(f"\nConfig:")
    print(f"  Ollama host:   {cfg.ollama_host}")
    print(f"  Embed model:   {cfg.embed_model}")
    print(f"  Chat model:    {cfg.chat_model}")
    print(f"  Chroma path:   {cfg.chroma_path}")
    print(f"  Collection:    {cfg.collection}")
    print(f"  Chunk size:    {cfg.chunk_size}")
    print(f"  Chunk overlap: {cfg.chunk_overlap}")
    print(f"  Top-K:         {cfg.top_k}")
    print()

    ollama_ok = check_ollama(cfg)

    try:
        chroma = get_chroma(cfg)
        col = get_collection(chroma, cfg)
        count = col.count()
        print(f"   Collection '{cfg.collection}': {count} chunks")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")

# ─── Tune ─────────────────────────────────────────────────────────────────────

def cmd_tune(args, cfg: Config):
    """Interactive tuning: vary chunk_size and top_k, score on a test question."""
    if not check_ollama(cfg):
        sys.exit(1)

    print("\n🎛  Interactive Tuning Mode")
    print("   This will re-ingest your documents with different settings and")
    print("   measure answer quality on a test question.\n")

    source = input("Path to your documents (dir or file): ").strip()
    question = input("Test question to evaluate with: ").strip()
    expected = input("Expected keyword(s) in a good answer (comma-separated): ").strip()
    expected_kws = [k.strip().lower() for k in expected.split(",") if k.strip()]

    configs_to_try = [
        {"chunk_size": 256,  "chunk_overlap": 32,  "top_k": 3},
        {"chunk_size": 512,  "chunk_overlap": 64,  "top_k": 5},
        {"chunk_size": 1024, "chunk_overlap": 128, "top_k": 5},
        {"chunk_size": 512,  "chunk_overlap": 64,  "top_k": 10},
    ]

    results = []

    for params in configs_to_try:
        label = f"chunk={params['chunk_size']} overlap={params['chunk_overlap']} top_k={params['top_k']}"
        print(f"\n🔬 Testing: {label}")

        # Temp config
        test_cfg = Config(
            ollama_host=cfg.ollama_host,
            embed_model=cfg.embed_model,
            chat_model=cfg.chat_model,
            chroma_path=f"./chroma_tune_{params['chunk_size']}_{params['top_k']}",
            collection="tune_test",
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            top_k=params["top_k"],
        )

        # Ingest
        chroma = get_chroma(test_cfg)
        col = get_collection(chroma, test_cfg)
        src_path = Path(source)
        files = list(src_path.rglob("*")) if src_path.is_dir() else [src_path]
        for fpath in [f for f in files if f.is_file()]:
            text = load_file(fpath)
            if not text:
                continue
            chunks = chunk_text(text, test_cfg.chunk_size, test_cfg.chunk_overlap)
            ids = [hashlib.md5(f"{fpath.name}::{i}".encode()).hexdigest() for i in range(len(chunks))]
            embeddings = [embed(c, test_cfg) for c in chunks]
            col.upsert(ids=ids, embeddings=embeddings, documents=chunks,
                       metadatas=[{"source": fpath.name, "chunk_index": i} for i in range(len(chunks))])

        # Query
        t0 = time.time()
        q_emb = embed(question, test_cfg)
        res = col.query(query_embeddings=[q_emb], n_results=min(test_cfg.top_k, col.count()))
        context = "\n\n---\n\n".join(res["documents"][0])

        client = ollama_lib.Client(host=test_cfg.ollama_host)
        response = client.chat(
            model=test_cfg.chat_model,
            messages=[
                {"role": "system", "content": "Answer using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        elapsed = time.time() - t0
        answer = response["message"]["content"]

        # Score
        score = sum(1 for kw in expected_kws if kw in answer.lower()) / max(len(expected_kws), 1)
        results.append({"label": label, "score": score, "latency": elapsed, "answer_preview": answer[:120]})
        print(f"   Score: {score:.0%} | Latency: {elapsed:.1f}s")
        print(f"   Answer: {answer[:120]}...")

    print("\n" + "=" * 60)
    print("  Tuning Results (ranked by score, then latency)")
    print("=" * 60)
    ranked = sorted(results, key=lambda r: (-r["score"], r["latency"]))
    for i, r in enumerate(ranked):
        star = " ⭐" if i == 0 else ""
        print(f"  {i+1}. {r['label']}{star}")
        print(f"     Score: {r['score']:.0%} | Latency: {r['latency']:.1f}s")

    best = ranked[0]
    print(f"\n💡 Recommended settings: {best['label']}")

# ─── Eval ─────────────────────────────────────────────────────────────────────

def cmd_eval(args, cfg: Config):
    """Run a JSON eval set and report keyword recall scores."""
    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"❌ Eval file not found: {eval_path}")
        sys.exit(1)

    evals = json.loads(eval_path.read_text())
    if not check_ollama(cfg):
        sys.exit(1)

    chroma = get_chroma(cfg)
    col = get_collection(chroma, cfg)

    if col.count() == 0:
        print("❌ Collection is empty. Ingest documents first.")
        sys.exit(1)

    print(f"\n🧪 Running {len(evals)} eval cases...\n")
    total_score = 0.0

    for i, case in enumerate(evals):
        question = case["question"]
        expected_kws = [k.lower() for k in case.get("expected_keywords", [])]

        q_emb = embed(question, cfg)
        results = col.query(query_embeddings=[q_emb], n_results=min(cfg.top_k, col.count()))
        context = "\n\n---\n\n".join(results["documents"][0])

        client = ollama_lib.Client(host=cfg.ollama_host)
        response = client.chat(
            model=cfg.chat_model,
            messages=[
                {"role": "system", "content": "Answer using only the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        answer = response["message"]["content"]
        score = sum(1 for kw in expected_kws if kw in answer.lower()) / max(len(expected_kws), 1)
        total_score += score

        status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
        print(f"  {status} [{i+1}/{len(evals)}] {question[:60]}")
        print(f"     Score: {score:.0%} | Keywords hit: {sum(1 for kw in expected_kws if kw in answer.lower())}/{len(expected_kws)}")

    avg = total_score / max(len(evals), 1)
    print(f"\n{'='*50}")
    print(f"  Average keyword recall: {avg:.0%} across {len(evals)} questions")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ollama + ChromaDB RAG for Raspberry Pi")
    parser.add_argument("--ollama-host",   default=None, help="Override OLLAMA_HOST")
    parser.add_argument("--embed-model",   default=None, help="Override EMBED_MODEL")
    parser.add_argument("--chat-model",    default=None, help="Override CHAT_MODEL")
    parser.add_argument("--collection",    default=None, help="Override COLLECTION")
    parser.add_argument("--chunk-size",    type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--top-k",         type=int, default=None)

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents into ChromaDB")
    p_ingest.add_argument("source", help="File or directory to ingest")

    p_query = sub.add_parser("query", help="Ask a question via RAG")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--show-chunks", action="store_true", help="Print retrieved chunks")

    sub.add_parser("status", help="Check connections and collection stats")
    sub.add_parser("tune",   help="Interactive tuning session")

    p_eval = sub.add_parser("eval", help="Run eval set from JSON file")
    p_eval.add_argument("eval_file", help="Path to evals JSON file")

    args = parser.parse_args()

    # Apply CLI overrides to config
    if args.ollama_host:   CFG.ollama_host   = args.ollama_host
    if args.embed_model:   CFG.embed_model   = args.embed_model
    if args.chat_model:    CFG.chat_model    = args.chat_model
    if args.collection:    CFG.collection    = args.collection
    if args.chunk_size:    CFG.chunk_size    = args.chunk_size
    if args.chunk_overlap: CFG.chunk_overlap = args.chunk_overlap
    if args.top_k:         CFG.top_k         = args.top_k

    dispatch = {
        "ingest": cmd_ingest,
        "query":  cmd_query,
        "status": cmd_status,
        "tune":   cmd_tune,
        "eval":   cmd_eval,
    }
    dispatch[args.cmd](args, CFG)

if __name__ == "__main__":
    main()
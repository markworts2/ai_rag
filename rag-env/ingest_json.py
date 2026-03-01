import json
import chromadb
import argparse
import os
import uuid
import requests
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def get_ollama_embedding(text: str, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434") -> list[float]:
    """Get embedding from local Ollama instance."""
    response = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": model, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_ollama_embeddings_batch(texts: list[str], model: str, ollama_url: str) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    embeddings = []
    for i, text in enumerate(texts):
        print(f"  Embedding {i + 1}/{len(texts)}...", end="\r")
        embeddings.append(get_ollama_embedding(text, model, ollama_url))
    print()
    return embeddings


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB embedding function using Ollama."""
    def __init__(self, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self._verify_connection()

    def _verify_connection(self):
        """Check Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            available = [m["name"] for m in response.json().get("models", [])]
            matches = [m for m in available if m.startswith(self.model)]
            if not matches:
                print(f"Warning: '{self.model}' not found in Ollama.")
                print(f"Available models: {available}")
                print(f"Pull it with: ollama pull {self.model}")
            else:
                print(f"Ollama connected. Using model: {matches[0]}")
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            )

    def __call__(self, input: list[str]) -> list[list[float]]:
        return get_ollama_embeddings_batch(input, self.model, self.ollama_url)


def ingest_json_to_chroma(
    json_file_path: str,
    collection_name: str = "rag_docs",
    chroma_db_path: str = "./chroma_db",
    text_field: str = None,

    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
):
    # Load JSON
    print(f"Loading JSON from: {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = [data] if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("JSON must be a list of objects or a single object.")
    print(f"Found {len(records)} records.")

    # Init ChromaDB with Ollama embedding function
    client = chromadb.PersistentClient(path=chroma_db_path)
    ef = OllamaEmbeddingFunction(model=embedding_model, ollama_url=ollama_url)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    # Auto-detect text field
    if text_field is None:
        sample = records[0] if records else {}
        candidates = ["text", "content", "description", "body", "title", "name", "message"]
        text_field = next((c for c in candidates if c in sample), None)
        if text_field:
            print(f"Auto-detected text field: '{text_field}'")
        else:
            print("No text field detected — serializing entire record as document text.")

    # Prepare data
    documents, metadatas, ids = [], [], []

    for record in records:
        if text_field and text_field in record:
            doc_text = str(record[text_field])
            metadata = {
                k: str(v)
                for k, v in record.items()
                if k != text_field and v is not None
            }
        else:
            doc_text = json.dumps(record)
            metadata = {}

        if not metadata:
            metadata = {"_source": os.path.basename(json_file_path)}

        doc_id = str(record.get("id", record.get("_id", str(uuid.uuid4()))))
        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(doc_id)

    # Upsert in batches
    batch_size = 50  # smaller batches since Ollama is slower than sentence-transformers
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        print(f"Processing records {start + 1}–{min(end, len(documents))}...")
        collection.upsert(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)

    print(f"\nDone! Collection '{collection_name}' now has {collection.count()} documents.")
    print(f"ChromaDB persisted at: {os.path.abspath(chroma_db_path)}")


def query_collection(chroma_db_path: str, collection_name: str, query: str,
                     n_results: int = 5, ollama_url: str = "http://localhost:11434",
                     embedding_model: str = "nomic-embed-text"):
    client = chromadb.PersistentClient(path=chroma_db_path)
    ef = OllamaEmbeddingFunction(model=embedding_model, ollama_url=ollama_url)
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a JSON file into ChromaDB using Ollama embeddings")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("--collection", default="my_collection", help="Collection name")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB persist path")
    parser.add_argument("--text-field", default=None, help="JSON field to use as document text")
    parser.add_argument("--ollama-url", default="http://192.168.1.169:11434", help="Ollama base URL")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--query", default=None, help="Optional: run a test query after ingestion")
    parser.add_argument("--query-n", default=5, type=int, help="Number of query results to return")

    args = parser.parse_args()

    ingest_json_to_chroma(
        json_file_path=args.json_file,
        collection_name=args.collection,
        chroma_db_path=args.db_path,
        text_field=args.text_field,
        ollama_url=args.ollama_url,
        embedding_model=args.model,
    )

    if args.query:
        print(f"\nRunning test query: '{args.query}'")
        results = query_collection(args.db_path, args.collection, args.query,
                                   n_results=args.query_n, ollama_url=args.ollama_url,
                                   embedding_model=args.model)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"\n--- Result ---\n{doc}\nMetadata: {meta}")
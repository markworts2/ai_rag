import json
import chromadb
from chromadb.utils import embedding_functions
import argparse
import os
import uuid

def ingest_json_to_chroma(
    json_file_path: str,
    collection_name: str = "my_collection",
    chroma_db_path: str = "./chroma_db",
    text_field: str = None,
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """
    Ingest a JSON file into ChromaDB.
    
    Args:
        json_file_path: Path to the JSON file
        collection_name: Name of the ChromaDB collection
        chroma_db_path: Path to persist ChromaDB
        text_field: Field to use as document text (auto-detected if None)
        embedding_model: Sentence transformer model for embeddings
    """
    # Load JSON
    print(f"Loading JSON from: {json_file_path}")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to list
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("JSON must be a list of objects or a single object.")

    print(f"Found {len(records)} records.")

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chroma_db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=ef)

    # Auto-detect text field if not provided
    if text_field is None:
        sample = records[0] if records else {}
        candidates = ["text", "content", "description", "body", "title", "name", "message"]
        text_field = next((c for c in candidates if c in sample), None)
        if text_field is None:
            # Fall back to serializing entire record as text
            print("No text field detected — serializing entire record as document text.")

    # Prepare batches
    documents, metadatas, ids = [], [], []

    for i, record in enumerate(records):
        if text_field and text_field in record:
            doc_text = str(record[text_field])
            print(doc_text)
            metadata = {k: str(v) for k, v in record.items() if k != text_field}
            print(metadata)
        else:
            doc_text = json.dumps(record)
            metadata = {}

        # Use existing id field or generate one
        doc_id = str(record.get("id", record.get("_id", str(uuid.uuid4()))))

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(doc_id)

    # Upsert in batches of 500
    batch_size = 500
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"Upserted records {start + 1}–{min(end, len(documents))}")

    print(f"\nDone! Collection '{collection_name}' now has {collection.count()} documents.")
    print(f"ChromaDB persisted at: {os.path.abspath(chroma_db_path)}")


def query_collection(chroma_db_path: str, collection_name: str, query: str, n_results: int = 5):
    """Quick helper to query the collection after ingestion."""
    client = chromadb.PersistentClient(path=chroma_db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction()
    collection = client.get_collection(name=collection_name, embedding_function=ef)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a JSON file into ChromaDB")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("--collection", default="my_collection", help="Collection name")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB persist path")
    parser.add_argument("--text-field", default=None, help="JSON field to use as document text")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--query", default=None, help="Optional: run a test query after ingestion")

    args = parser.parse_args()

    ingest_json_to_chroma(
        json_file_path=args.json_file,
        collection_name=args.collection,
        chroma_db_path=args.db_path,
        text_field=args.text_field,
        embedding_model=args.model,
    )

    if args.query:
        print(f"\nRunning test query: '{args.query}'")
        results = query_collection(args.db_path, args.collection, args.query)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"\n--- Result ---\n{doc}\nMetadata: {meta}")
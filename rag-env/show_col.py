import chromadb
import ollama as ollama_lib
results = collection.get(limit=5)
print(results["documents"])
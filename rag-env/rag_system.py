import chromadb
from sentence_transformers import SentenceTransformer
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

class RAGSystem:
    def __init__(self):
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector database
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    
    def add_documents(self, texts):
        # Split texts into chunks
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
        
        # Generate embeddings
        embeddings = self.embedder.encode(chunks).tolist()
        
        # Add to vector database
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"doc_{i}" for i in range(len(chunks))]
        )
    
    def retrieve(self, query, n_results=3):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results['documents'][0]
    
    def generate_response(self, query, context):
        prompt = f"""
        Context: {' '.join(context)}
        
        Question: {query}
        
        Answer based on the context:"""
        
        # Call Ollama API
        response = requests.post('http://localhost:11434/api/generate',
            json={
                'model': 'llama2:7b-chat-q4_0',
                'prompt': prompt,
                'stream': False
            })
        
        return response.json()['response']
    
    def query(self, question):
        context = self.retrieve(question)
        response = self.generate_response(question, context)
        return response

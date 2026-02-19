import streamlit as st
from rag_system import RAGSystem

st.title("Raspberry Pi RAG System")

# Initialize RAG system
@st.cache_resource
def load_rag_system():
    return RAGSystem()

rag = load_rag_system()

# File upload
uploaded_files = st.file_uploader(
    "Upload documents", 
    type=['txt', 'pdf'], 
    accept_multiple_files=True
)

if uploaded_files:
    documents = []
    for file in uploaded_files:
        content = file.read().decode('utf-8')
        documents.append(content)
    
    if st.button("Add Documents"):
        rag.add_documents(documents)
        st.success("Documents added successfully!")

# Query interface
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Generating response..."):
        response = rag.query(query)
        st.write(response)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize model and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # Embedding dimension is 384
documents = []  # Store documents and metadata

# Add text to vector database
def add_to_vector_db(text, metadata):
    embedding = model.encode([text])
    index.add(embedding)
    documents.append({"text": text, "metadata": metadata})

# Search the vector database
def search_vector_db(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [{"text": documents[i]["text"], "metadata": documents[i]["metadata"]} for i in indices[0]]
    return results

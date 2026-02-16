# gpt_memory.py
import faiss
import numpy as np
from openai import OpenAI
from config import API_KEY, EMBEDDING_MODEL, TOP_K_EMBEDDING

# OpenAI client
client = OpenAI(api_key=API_KEY)

class GPTMemory:
    def __init__(self):
        self.responses = []          # Original text chunks
        self.embeddings = None       # Embedding vectors
        self.index = None            # FAISS index

    def add_response(self, text):
        # Add new text
        self.responses.append(text)

        # Generate embeddings for all responses
        self.embeddings = np.array([
            self.get_embedding(t) for t in self.responses
        ]).astype("float32")

        # Build/rebuild FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def get_embedding(self, text):
        """Get embedding vector using OpenAI text-embedding-3-small"""
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)

    def retrieve_topk(self, query, top_k=TOP_K_EMBEDDING):
        if not self.responses:
            return []

        # Embed query
        query_vec = self.get_embedding(query).reshape(1, -1).astype("float32")

        # Search FAISS
        distances, indices = self.index.search(query_vec, top_k)
        return [self.responses[i] for i in indices[0]]

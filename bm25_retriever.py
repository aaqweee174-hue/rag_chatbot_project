# bm25_retriever.py
from rank_bm25 import BM25Okapi
import re

class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

    def _tokenize(self, text: str):
        # simple clean tokenizer
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def add_documents(self, docs: list[str]):
        """
        Add document chunks for BM25 indexing
        """
        self.documents.extend(docs)
        self.tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, top_k: int = 3):
        """
        Keyword-based retrieval
        """
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [self.documents[i] for i in top_indices]

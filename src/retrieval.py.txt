from typing import List, Dict
import numpy as np
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """
    Handles semantic retrieval using embeddings + FAISS.
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore, metadata: List[Dict]):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.metadata = metadata

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k relevant chunks for a given query.
        """
        query_embedding = self.embedding_model.encode_texts([query])

        scores, indices = self.vector_store.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk_data = self.metadata[idx]
            results.append({
                "score": float(score),
                "document_name": chunk_data["document_name"],
                "page_number": chunk_data["page_number"],
                "chunk_text": chunk_data["chunk_text"]
            })

        return results
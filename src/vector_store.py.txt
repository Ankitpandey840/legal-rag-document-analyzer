import faiss
import numpy as np
from typing import List


class VectorStore:
    """
    Handles FAISS vector indexing and similarity search.
    """

    def __init__(self, dimension: int):
        # Using Inner Product for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to FAISS index.
        """
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Search top_k most similar vectors.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        return scores, indices

    def save(self, path: str):
        """
        Save FAISS index to disk.
        """
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """
        Load FAISS index from disk.
        """
        self.index = faiss.read_index(path)
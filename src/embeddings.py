from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Handles embedding generation using Sentence Transformers.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of texts.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
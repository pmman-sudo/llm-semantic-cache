
import numpy as np
from gptcache.embedding import Onnx


class EmbeddingModel:
    """Generate semantic embeddings using GPTCache's ONNX model."""

    def __init__(self):
        self.model = Onnx()
        self.dimension = self.model.dimension

    def embed(self, text: str) -> np.ndarray:
        """Convert text into a normalized float32 embedding."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        vector = self.model.to_embeddings(text)

        vector = np.asarray(vector, dtype=np.float32).reshape(-1)

        if vector.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} dimensions, got {vector.size}."
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Embedding vector has zero magnitude.")

        return vector / norm

    @staticmethod
    def similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vector_b, dtype=np.float32).reshape(-1)

        if a.shape != b.shape:
            raise ValueError("Embedding dimensions must match.")

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)

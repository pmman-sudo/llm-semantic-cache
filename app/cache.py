
import os
import sqlite3
import threading

import faiss
import numpy as np


class SemanticCache:
    """Semantic cache using FAISS for similarity search and SQLite for storage."""

    def __init__(
        self,
        dimension: int = 768,
        threshold: float = 0.55,
        db_path: str = "data/cache.db",
    ):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between -1 and 1.")

        self.dimension = dimension
        self.threshold = threshold
        self.db_path = db_path
        self.lock = threading.RLock()

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        self.connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                response TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.connection.commit()

        # Inner-product search on normalized vectors equals cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.ids = []

        self._load_index()

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if vector.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} dimensions, got {vector.size}."
            )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Embedding vector cannot be zero.")

        return np.ascontiguousarray(vector / norm)

    def _load_index(self):
        """Restore previously stored embeddings into FAISS."""
        rows = self.connection.execute(
            "SELECT id, embedding FROM cache ORDER BY id"
        ).fetchall()

        if not rows:
            return

        vectors = []
        for record_id, embedding_blob in rows:
            vector = np.frombuffer(
                embedding_blob, dtype=np.float32
            ).copy()

            vectors.append(self._validate_embedding(vector))
            self.ids.append(record_id)

        matrix = np.ascontiguousarray(
            np.vstack(vectors), dtype=np.float32
        )

        self.index.add(matrix)

    def lookup(self, embedding: np.ndarray):
        """Find the most similar cached response, if one meets the threshold."""
        vector = self._validate_embedding(embedding)

        with self.lock:
            if self.index.ntotal == 0:
                return None

            scores, positions = self.index.search(
                vector.reshape(1, -1), 1
            )

            similarity = float(scores[0][0])
            position = int(positions[0][0])

            if position < 0 or similarity < self.threshold:
                return None

            record_id = self.ids[position]

            row = self.connection.execute(
                "SELECT question, response FROM cache WHERE id = ?",
                (record_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "question": row[0],
                "response": row[1],
                "similarity": similarity,
            }

    def store(
        self,
        question: str,
        embedding: np.ndarray,
        response: str,
    ):
        """Persist a question and response and add its embedding to FAISS."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not response.strip():
            raise ValueError("Response cannot be empty.")

        vector = self._validate_embedding(embedding)

        with self.lock:
            cursor = self.connection.execute(
                """
                INSERT INTO cache (question, response, embedding)
                VALUES (?, ?, ?)
                """,
                (
                    question,
                    response,
                    vector.tobytes(),
                ),
            )

            self.connection.commit()

            self.index.add(
                vector.reshape(1, -1)
            )

            self.ids.append(cursor.lastrowid)

    def count(self) -> int:
        """Return the number of cached records."""
        with self.lock:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM cache"
            ).fetchone()

            return row[0]

    def close(self):
        """Close the SQLite connection."""
        with self.lock:
            self.connection.close()

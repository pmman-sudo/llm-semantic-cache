
import numpy as np
import pytest

from app.cache import SemanticCache


@pytest.fixture
def cache(tmp_path):
    """Create an isolated cache database for each test."""
    db_path = tmp_path / "test_cache.db"

    instance = SemanticCache(
        dimension=3,
        threshold=0.80,
        db_path=str(db_path),
    )

    yield instance

    instance.close()


def test_store_and_retrieve(cache):
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    cache.store(
        "What is Docker?",
        vector,
        "Docker runs applications in containers.",
    )

    result = cache.lookup(vector)

    assert result is not None
    assert result["question"] == "What is Docker?"
    assert result["response"] == "Docker runs applications in containers."


def test_similar_question_returns_cache_hit(cache):
    original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    similar = np.array([0.99, 0.10, 0.0], dtype=np.float32)

    cache.store("What is Docker?", original, "Docker uses containers.")

    result = cache.lookup(similar)

    assert result is not None
    assert result["similarity"] >= 0.80


def test_unrelated_question_returns_cache_miss(cache):
    original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    unrelated = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    cache.store("What is Docker?", original, "Docker uses containers.")

    result = cache.lookup(unrelated)

    assert result is None


def test_cache_survives_database_restart(tmp_path):
    db_path = str(tmp_path / "persistent_cache.db")
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    first_cache = SemanticCache(
        dimension=3,
        threshold=0.80,
        db_path=db_path,
    )

    first_cache.store(
        "What is Kubernetes?",
        vector,
        "Kubernetes orchestrates containers.",
    )

    first_cache.close()

    second_cache = SemanticCache(
        dimension=3,
        threshold=0.80,
        db_path=db_path,
    )

    try:
        result = second_cache.lookup(vector)

        assert result is not None
        assert result["response"] == "Kubernetes orchestrates containers."
        assert second_cache.count() == 1
    finally:
        second_cache.close()


def test_invalid_embedding_dimension_is_rejected(cache):
    invalid_vector = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Expected 3 dimensions"):
        cache.lookup(invalid_vector)


def test_zero_embedding_is_rejected(cache):
    zero_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="cannot be zero"):
        cache.lookup(zero_vector)


def test_empty_question_is_rejected(cache):
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        cache.store("", vector, "Some response.")


def test_cache_count(cache):
    assert cache.count() == 0

    cache.store(
        "What is Terraform?",
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Terraform manages infrastructure.",
    )

    assert cache.count() == 1

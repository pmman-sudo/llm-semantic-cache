
import time

from app.embeddings import EmbeddingModel
from app.cache import SemanticCache
from app.llm import MockLLM
from app.metrics import CacheMetrics


class SemanticCacheApp:
    """Integrate embeddings, semantic retrieval, LLM and metrics."""

    def __init__(
        self,
        threshold: float = 0.55,
        simulated_latency: float = 0.5,
        cost_per_llm_call: float = 0.002,
    ):
        self.embedding_model = EmbeddingModel()

        self.cache = SemanticCache(
            dimension=self.embedding_model.dimension,
            threshold=threshold,
        )

        self.llm = MockLLM(
            simulated_latency=simulated_latency
        )

        self.metrics = CacheMetrics(
            cost_per_llm_call=cost_per_llm_call
        )

    def ask(self, question: str) -> dict:
        """Answer a question using semantic caching."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()
        start_time = time.perf_counter()

        embedding = self.embedding_model.embed(question)

        cached_result = self.cache.lookup(embedding)

        if cached_result:
            answer = cached_result["response"]
            cache_hit = True
            similarity = cached_result["similarity"]
        else:
            answer = self.llm.generate(question)
            self.cache.store(question, embedding, answer)
            cache_hit = False
            similarity = None

        latency = time.perf_counter() - start_time

        self.metrics.record(
            cache_hit=cache_hit,
            latency=latency,
        )

        return {
            "question": question,
            "answer": answer,
            "cache_hit": cache_hit,
            "similarity": similarity,
            "latency_seconds": latency,
        }

    def show_metrics(self):
        """Display application performance statistics."""
        print("\n========== CACHE METRICS ==========")

        for key, value in self.metrics.summary().items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("Actual simulated LLM calls:", self.llm.call_count)
        print("Stored cache records:", self.cache.count())
        print("===================================\n")

    def close(self):
        self.cache.close()


def main():
    app = SemanticCacheApp()

    print("\n======================================")
    print("       LLM SEMANTIC CACHE SYSTEM")
    print("======================================")
    print("Ask a question to test semantic caching.")
    print("Type 'metrics' to view statistics.")
    print("Type 'exit' to quit.")

    try:
        while True:
            question = input("\nYour question: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if question.lower() == "metrics":
                app.show_metrics()
                continue

            if not question:
                print("Please enter a question.")
                continue

            try:
                result = app.ask(question)

                print("\n--- RESULT ---")
                print("Question:", result["question"])
                print(
                    "Cache status:",
                    "HIT" if result["cache_hit"] else "MISS",
                )

                if result["similarity"] is not None:
                    print(
                        "Similarity:",
                        round(result["similarity"], 4),
                    )

                print("Answer:", result["answer"])
                print(
                    "Latency:",
                    f"{result['latency_seconds']:.4f} seconds",
                )

            except Exception as error:
                print("Request failed:", error)

    except KeyboardInterrupt:
        print("\nApplication interrupted.")

    finally:
        app.show_metrics()
        app.close()
        print("Semantic cache application closed.")


if __name__ == "__main__":
    main()


import csv
import json
import os
import statistics
import tempfile
import time

import matplotlib.pyplot as plt

from app.embeddings import EmbeddingModel
from app.cache import SemanticCache
from app.llm import MockLLM
from app.metrics import CacheMetrics


SIMULATED_LATENCIES = (0.1, 0.5, 1.0, 2.0, 5.0)
COST_PER_CALL = 0.002
THRESHOLD = 0.55
TRIALS = 5


def load_workload():
    workload_path = os.path.join(
        os.path.dirname(__file__), "workload.json"
    )

    with open(workload_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark workload must be a non-empty list.")

    if not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError("Every workload item must be a non-empty string.")

    return questions


def run_baseline(questions, simulated_latency):
    """Run every question without semantic caching."""
    llm = MockLLM(simulated_latency=simulated_latency)
    start = time.perf_counter()

    for question in questions:
        llm.generate(question)

    return {
        "llm_calls": llm.call_count,
        "total_latency": time.perf_counter() - start,
    }


def run_cached(questions, db_path, embedding_model, simulated_latency):
    """Run the workload through a fresh semantic cache."""
    cache = SemanticCache(
        dimension=embedding_model.dimension,
        threshold=THRESHOLD,
        db_path=db_path,
    )

    llm = MockLLM(simulated_latency=simulated_latency)
    metrics = CacheMetrics(cost_per_llm_call=COST_PER_CALL)

    try:
        for question in questions:
            start = time.perf_counter()

            embedding = embedding_model.embed(question)
            cached_result = cache.lookup(embedding)

            if cached_result:
                cache_hit = True
            else:
                answer = llm.generate(question)
                cache.store(question, embedding, answer)
                cache_hit = False

            latency = time.perf_counter() - start
            metrics.record(cache_hit, latency)

        return {
            "llm_calls": llm.call_count,
            "total_latency": metrics.total_latency,
            "metrics": metrics.summary(),
        }

    finally:
        cache.close()


def main():
    questions = load_workload()
    embedding_model = EmbeddingModel()

    # Exclude initial model loading/warm-up from measured requests.
    embedding_model.embed("Latency sensitivity experiment warm-up.")

    os.makedirs("reports", exist_ok=True)

    all_results = []

    print("========== LATENCY SENSITIVITY EXPERIMENT ==========")
    print("Questions per trial:", len(questions))
    print("Trials per latency:", TRIALS)
    print("Cache threshold:", THRESHOLD)

    for simulated_latency in SIMULATED_LATENCIES:
        print(f"\nTesting simulated LLM latency: {simulated_latency}s")

        baseline_runs = []
        cached_runs = []

        for trial in range(1, TRIALS + 1):
            baseline = run_baseline(
                questions, simulated_latency
            )
            baseline_runs.append(baseline)

            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "latency.db")

                cached = run_cached(
                    questions,
                    db_path,
                    embedding_model,
                    simulated_latency,
                )

            cached_runs.append(cached)

            print(
                f"Trial {trial}/{TRIALS} | "
                f"Baseline: {baseline['total_latency']:.3f}s | "
                f"Cached: {cached['total_latency']:.3f}s | "
                f"Hits: {cached['metrics']['cache_hits']}"
            )

        baseline_totals = [
            run["total_latency"] for run in baseline_runs
        ]
        cached_totals = [
            run["total_latency"] for run in cached_runs
        ]

        baseline_average = statistics.mean(baseline_totals)
        cached_average = statistics.mean(cached_totals)

        improvement = (
            (baseline_average - cached_average)
            / baseline_average * 100
            if baseline_average else 0
        )

        average_hits = statistics.mean([
            run["metrics"]["cache_hits"] for run in cached_runs
        ])

        average_hit_rate = statistics.mean([
            run["metrics"]["hit_rate"] for run in cached_runs
        ])

        average_llm_calls = statistics.mean([
            run["llm_calls"] for run in cached_runs
        ])

        average_cost_saved = statistics.mean([
            run["metrics"]["estimated_cost_saved"]
            for run in cached_runs
        ])

        result = {
            "simulated_llm_latency_seconds": simulated_latency,
            "trials": TRIALS,
            "questions_per_trial": len(questions),
            "baseline_average_total_seconds": round(
                baseline_average, 4
            ),
            "baseline_median_total_seconds": round(
                statistics.median(baseline_totals), 4
            ),
            "cached_average_total_seconds": round(
                cached_average, 4
            ),
            "cached_median_total_seconds": round(
                statistics.median(cached_totals), 4
            ),
            "average_latency_improvement_percent": round(
                improvement, 2
            ),
            "average_cache_hits": round(average_hits, 2),
            "average_cache_hit_rate_percent": round(
                average_hit_rate, 2
            ),
            "average_llm_calls": round(average_llm_calls, 2),
            "average_estimated_cost_saved": round(
                average_cost_saved, 4
            ),
            "individual_trials": [
                {
                    "trial": index + 1,
                    "baseline_seconds": round(
                        baseline_totals[index], 4
                    ),
                    "cached_seconds": round(
                        cached_totals[index], 4
                    ),
                    "cache_hits": cached_runs[index]["metrics"][
                        "cache_hits"
                    ],
                }
                for index in range(TRIALS)
            ],
        }

        all_results.append(result)

        print(
            f"Average improvement: {improvement:.2f}% | "
            f"Average cache hit rate: {average_hit_rate:.2f}%"
        )

    # Export summary CSV.
    csv_path = "reports/latency_sensitivity.csv"

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = [
            "simulated_llm_latency_seconds",
            "baseline_average_total_seconds",
            "cached_average_total_seconds",
            "average_latency_improvement_percent",
            "average_cache_hits",
            "average_cache_hit_rate_percent",
            "average_llm_calls",
            "average_estimated_cost_saved",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                field: result[field]
                for field in fieldnames
            })

    # Export detailed JSON results.
    json_path = "reports/latency_sensitivity.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "methodology": {
                    "workload_size": len(questions),
                    "trials_per_latency": TRIALS,
                    "latency_values_seconds": SIMULATED_LATENCIES,
                    "cache_threshold": THRESHOLD,
                    "cost_per_llm_call": COST_PER_CALL,
                    "embedding_warmup_excluded": True,
                    "fresh_cache_per_trial": True,
                    "mock_llm": True,
                },
                "results": all_results,
            },
            file,
            indent=4,
        )

    latencies = [
        result["simulated_llm_latency_seconds"]
        for result in all_results
    ]

    baseline_averages = [
        result["baseline_average_total_seconds"]
        for result in all_results
    ]

    cached_averages = [
        result["cached_average_total_seconds"]
        for result in all_results
    ]

    improvements = [
        result["average_latency_improvement_percent"]
        for result in all_results
    ]

    # Chart 1: baseline versus cached execution.
    plt.figure(figsize=(9, 5))
    plt.plot(
        latencies, baseline_averages,
        marker="o", label="Without semantic cache"
    )
    plt.plot(
        latencies, cached_averages,
        marker="o", label="With semantic cache"
    )
    plt.xlabel("Simulated LLM latency (seconds)")
    plt.ylabel("Average total execution time (seconds)")
    plt.title("Latency Sensitivity: Baseline vs Semantic Cache")
    plt.xticks(latencies)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/latency_sensitivity.png", dpi=150)
    plt.close()

    # Chart 2: latency improvement across settings.
    plt.figure(figsize=(9, 5))
    plt.plot(
        latencies, improvements,
        marker="o"
    )
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Simulated LLM latency (seconds)")
    plt.ylabel("Latency improvement (%)")
    plt.title("Semantic Cache Latency Improvement")
    plt.xticks(latencies)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reports/latency_improvement.png", dpi=150)
    plt.close()

    print("\n========== FINAL RESULTS ==========")

    for result in all_results:
        print(
            f"{result['simulated_llm_latency_seconds']}s LLM delay | "
            f"Baseline: {result['baseline_average_total_seconds']}s | "
            f"Cached: {result['cached_average_total_seconds']}s | "
            f"Improvement: "
            f"{result['average_latency_improvement_percent']}%"
        )

    print("\nReports saved:")
    print(csv_path)
    print(json_path)
    print("reports/latency_sensitivity.png")
    print("reports/latency_improvement.png")


if __name__ == "__main__":
    main()

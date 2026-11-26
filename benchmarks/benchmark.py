
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


SIMULATED_LATENCY = 1.0
COST_PER_CALL = 0.002
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


def run_baseline(questions):
    """Run every question without semantic caching."""
    llm = MockLLM(simulated_latency=SIMULATED_LATENCY)
    latencies = []

    for question in questions:
        start = time.perf_counter()
        llm.generate(question)
        latencies.append(time.perf_counter() - start)

    return {
        "llm_calls": llm.call_count,
        "total_latency": sum(latencies),
        "average_latency": statistics.mean(latencies),
        "latencies": latencies,
    }


def run_cached(questions, db_path):
    """Run the workload through a fresh semantic cache."""
    embedding_model = EmbeddingModel()

    # Warm up the embedding model outside the measured workload.
    embedding_model.embed("Semantic cache benchmark warm-up.")

    cache = SemanticCache(
        dimension=embedding_model.dimension,
        threshold=0.55,
        db_path=db_path,
    )

    llm = MockLLM(simulated_latency=SIMULATED_LATENCY)
    metrics = CacheMetrics(cost_per_llm_call=COST_PER_CALL)
    results = []

    try:
        for question in questions:
            start = time.perf_counter()

            embedding = embedding_model.embed(question)
            cached_result = cache.lookup(embedding)

            if cached_result:
                answer = cached_result["response"]
                cache_hit = True
                similarity = cached_result["similarity"]
            else:
                answer = llm.generate(question)
                cache.store(question, embedding, answer)
                cache_hit = False
                similarity = None

            latency = time.perf_counter() - start
            metrics.record(cache_hit, latency)

            results.append({
                "question": question,
                "cache_status": "HIT" if cache_hit else "MISS",
                "similarity": (
                    round(similarity, 4)
                    if similarity is not None else ""
                ),
                "latency_seconds": round(latency, 6),
            })

        return {
            "llm_calls": llm.call_count,
            "total_latency": metrics.total_latency,
            "average_latency": metrics.average_latency,
            "metrics": metrics.summary(),
            "results": results,
        }

    finally:
        cache.close()


def average(values):
    return statistics.mean(values)


def median(values):
    return statistics.median(values)


def main():
    questions = load_workload()
    baseline_runs = []
    cached_runs = []

    print(f"Running {TRIALS} benchmark trials...")
    print(f"Questions per trial: {len(questions)}")
    print(f"Simulated LLM latency: {SIMULATED_LATENCY}s")
    print("Each trial starts with an empty cache.")

    for trial in range(1, TRIALS + 1):
        print(f"\nTrial {trial}/{TRIALS}")

        baseline = run_baseline(questions)
        baseline_runs.append(baseline)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "benchmark.db")
            cached = run_cached(questions, db_path)
            cached_runs.append(cached)

        print(
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

    baseline_avg = average(baseline_totals)
    cached_avg = average(cached_totals)

    time_saved = baseline_avg - cached_avg
    improvement = (
        time_saved / baseline_avg * 100
        if baseline_avg else 0
    )

    avg_hits = average([
        run["metrics"]["cache_hits"] for run in cached_runs
    ])
    avg_misses = average([
        run["metrics"]["cache_misses"] for run in cached_runs
    ])
    avg_hit_rate = average([
        run["metrics"]["hit_rate"] for run in cached_runs
    ])
    avg_cost_saved = average([
        run["metrics"]["estimated_cost_saved"]
        for run in cached_runs
    ])

    summary = {
        "trials": TRIALS,
        "workload_size_per_trial": len(questions),
        "total_requests_across_trials": len(questions) * TRIALS,
        "simulated_llm_latency_seconds": SIMULATED_LATENCY,
        "assumed_cost_per_llm_call": COST_PER_CALL,
        "baseline": {
            "average_total_latency_seconds": round(
                baseline_avg, 4
            ),
            "median_total_latency_seconds": round(
                median(baseline_totals), 4
            ),
            "average_latency_per_request_seconds": round(
                average([
                    run["average_latency"] for run in baseline_runs
                ]), 4
            ),
            "llm_calls_per_trial": len(questions),
        },
        "semantic_cache": {
            "average_total_latency_seconds": round(
                cached_avg, 4
            ),
            "median_total_latency_seconds": round(
                median(cached_totals), 4
            ),
            "average_latency_per_request_seconds": round(
                average([
                    run["average_latency"] for run in cached_runs
                ]), 4
            ),
            "average_llm_calls_per_trial": round(
                average([
                    run["llm_calls"] for run in cached_runs
                ]), 2
            ),
            "average_cache_hits_per_trial": round(avg_hits, 2),
            "average_cache_misses_per_trial": round(avg_misses, 2),
            "average_hit_rate_percent": round(avg_hit_rate, 2),
            "average_estimated_cost_saved": round(
                avg_cost_saved, 4
            ),
        },
        "average_time_saved_seconds": round(time_saved, 4),
        "average_latency_improvement_percent": round(
            improvement, 2
        ),
        "individual_trials": [
            {
                "trial": index + 1,
                "baseline_total_latency_seconds": round(
                    baseline_runs[index]["total_latency"], 4
                ),
                "cached_total_latency_seconds": round(
                    cached_runs[index]["total_latency"], 4
                ),
                "cache_hits": cached_runs[index]["metrics"]["cache_hits"],
                "cache_misses": cached_runs[index]["metrics"]["cache_misses"],
            }
            for index in range(TRIALS)
        ],
        "methodology_note": (
            "The LLM is simulated. Embedding warm-up is excluded "
            "from measured cached request latency. Each trial uses "
            "a fresh empty cache. Results are from a small synthetic "
            "workload and are not production performance guarantees."
        ),
    }

    os.makedirs("reports", exist_ok=True)

    # Save individual request measurements from every trial.
    with open(
        "reports/benchmark_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "trial",
            "question",
            "cache_status",
            "similarity",
            "latency_seconds",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for trial_index, run in enumerate(cached_runs, start=1):
            for result in run["results"]:
                writer.writerow({
                    "trial": trial_index,
                    **result,
                })

    with open(
        "reports/benchmark_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    # Chart 1: average total execution time.
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Without cache", "With semantic cache"],
        [baseline_avg, cached_avg],
    )
    plt.ylabel("Average total execution time (seconds)")
    plt.title(f"Average Execution Time Across {TRIALS} Trials")
    plt.tight_layout()
    plt.savefig("reports/benchmark_latency.png", dpi=150)
    plt.close()

    # Chart 2: average cache hits and misses per trial.
    plt.figure(figsize=(8, 5))
    plt.bar(
        ["Average cache hits", "Average cache misses"],
        [avg_hits, avg_misses],
    )
    plt.ylabel("Requests per trial")
    plt.title("Average Semantic Cache Performance")
    plt.tight_layout()
    plt.savefig("reports/cache_performance.png", dpi=150)
    plt.close()

    print("\n========== FINAL BENCHMARK RESULTS ==========")
    print("Trials:", TRIALS)
    print("Questions per trial:", len(questions))
    print("Average baseline latency:", round(baseline_avg, 4), "seconds")
    print("Median baseline latency:", round(median(baseline_totals), 4), "seconds")
    print("Average cached latency:", round(cached_avg, 4), "seconds")
    print("Median cached latency:", round(median(cached_totals), 4), "seconds")
    print("Average cache hits:", round(avg_hits, 2))
    print("Average cache misses:", round(avg_misses, 2))
    print("Average cache hit rate:", round(avg_hit_rate, 2), "%")
    print("Average LLM calls avoided:", round(len(questions) - average([
        run["llm_calls"] for run in cached_runs
    ]), 2))
    print("Average latency improvement:", round(improvement, 2), "%")
    print("Average estimated cost saved: $", round(avg_cost_saved, 4), sep="")
    print("=============================================")

    print("\nReports saved:")
    print("reports/benchmark_results.csv")
    print("reports/benchmark_summary.json")
    print("reports/benchmark_latency.png")
    print("reports/cache_performance.png")


if __name__ == "__main__":
    main()


import csv
import json
from pathlib import Path

import numpy as np

from app.cache import SemanticCache
from app.embeddings import EmbeddingModel


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA = ROOT / "data"

THRESHOLDS = np.round(np.arange(0.40, 0.901, 0.05), 2)


# Reference questions stored in the cache.
# Three examples per topic.
SEEDS = [
    ("What is Docker?", "docker"),
    ("Explain Docker containers.", "docker"),
    ("How does Docker package applications?", "docker"),

    ("What is Kubernetes?", "kubernetes"),
    ("Explain Kubernetes orchestration.", "kubernetes"),
    ("How does Kubernetes manage workloads?", "kubernetes"),

    ("What is Terraform?", "terraform"),
    ("Explain infrastructure as code with Terraform.", "terraform"),
    ("How does Terraform provision infrastructure?", "terraform"),

    ("What is AWS?", "aws"),
    ("Explain Amazon Web Services.", "aws"),
    ("What services does AWS provide?", "aws"),

    ("What is Python programming?", "python"),
    ("Explain the Python programming language.", "python"),
    ("What is Python used for?", "python"),
]

# Held-out questions: none of these are stored in the cache.
# Six evaluation questions per topic, plus ten unrelated questions.
EVALUATION_QUESTIONS = [
    ("How do containers package applications?", "docker"),
    ("What does Docker do?", "docker"),
    ("How can an application run consistently across environments?", "docker"),
    ("What is the role of container images?", "docker"),
    ("How are isolated application environments created?", "docker"),
    ("Why use containers for application deployment?", "docker"),

    ("How does Kubernetes manage containerized workloads?", "kubernetes"),
    ("What is container orchestration?", "kubernetes"),
    ("How can an application scale across multiple nodes?", "kubernetes"),
    ("What manages automated container deployment?", "kubernetes"),
    ("How does Kubernetes maintain application availability?", "kubernetes"),
    ("What is a Kubernetes cluster used for?", "kubernetes"),

    ("How can infrastructure be provisioned as code?", "terraform"),
    ("What problem does Terraform solve?", "terraform"),
    ("How can cloud resources be created automatically?", "terraform"),
    ("What is declarative infrastructure management?", "terraform"),
    ("How can infrastructure changes be tracked?", "terraform"),
    ("Why use configuration files to manage infrastructure?", "terraform"),

    ("Which services does Amazon Web Services provide?", "aws"),
    ("What is cloud computing?", "aws"),
    ("How can businesses rent computing resources online?", "aws"),
    ("What is an EC2 instance?", "aws"),
    ("Where can applications store files in AWS?", "aws"),
    ("What is the purpose of cloud infrastructure?", "aws"),

    ("What language is commonly used for automation and data analysis?", "python"),
    ("What is Python commonly used for?", "python"),
    ("Which programming language has a simple readable syntax?", "python"),
    ("How can scripts automate repetitive tasks?", "python"),
    ("What language is popular for machine learning?", "python"),
    ("Which language supports general-purpose application development?", "python"),

    ("How do relational databases store records?", "other"),
    ("What is the purpose of version control?", "other"),
    ("How does DNS resolve domain names?", "other"),
    ("What is the difference between HTTP and HTTPS?", "other"),
    ("Explain how a Linux process works.", "other"),
    ("What is the function of a computer network router?", "other"),
    ("How does public key cryptography work?", "other"),
    ("What is the purpose of an operating system kernel?", "other"),
    ("How do SQL queries retrieve database records?", "other"),
    ("What is the difference between RAM and storage?", "other"),
]


def get_topic(question):
    """Assign a topic label using the same categories as the evaluation."""
    text = question.lower()

    if "docker" in text or "container" in text:
        return "docker"
    if "kubernetes" in text or "k8s" in text:
        return "kubernetes"
    if "terraform" in text:
        return "terraform"
    if "aws" in text or "cloud" in text:
        return "aws"
    if "python" in text:
        return "python"

    return "other"


def evaluate_threshold(model, threshold):
    db_path = DATA / f"threshold_{threshold:.2f}.db"
    if db_path.exists():
        db_path.unlink()

    cache = SemanticCache(
        dimension=model.dimension,
        threshold=float(threshold),
        db_path=str(db_path),
    )

    correct_hits = 0
    incorrect_hits = 0
    misses = 0

    try:
        for question, topic in SEEDS:
            cache.store(
                question,
                model.embed(question),
                f"Reference response for {topic}.",
            )

        for question, expected_topic in EVALUATION_QUESTIONS:
            result = cache.lookup(model.embed(question))

            if result is None:
                misses += 1
                continue

            retrieved_topic = get_topic(result["question"])

            if retrieved_topic == expected_topic:
                correct_hits += 1
            else:
                incorrect_hits += 1

        total_hits = correct_hits + incorrect_hits
        total_queries = len(EVALUATION_QUESTIONS)

        precision = (
            correct_hits / total_hits if total_hits else 0.0
        )
        incorrect_hit_rate = (
            incorrect_hits / total_hits if total_hits else 0.0
        )
        coverage = total_hits / total_queries

        return {
            "threshold": float(threshold),
            "queries": total_queries,
            "correct_hits": correct_hits,
            "incorrect_hits": incorrect_hits,
            "misses": misses,
            "precision": round(precision, 4),
            "incorrect_hit_rate": round(incorrect_hit_rate, 4),
            "coverage": round(coverage, 4),
        }

    finally:
        cache.close()
        if db_path.exists():
            db_path.unlink()


def main():
    REPORTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    print("Loading embedding model...")
    model = EmbeddingModel()

    results = []

    print("\nRunning similarity threshold evaluation...")

    for threshold in THRESHOLDS:
        result = evaluate_threshold(model, threshold)
        results.append(result)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Correct: {result['correct_hits']} | "
            f"Incorrect: {result['incorrect_hits']} | "
            f"Misses: {result['misses']} | "
            f"Precision: {result['precision']:.2%} | "
            f"Coverage: {result['coverage']:.2%}"
        )

    csv_path = REPORTS / "threshold_evaluation.csv"
    json_path = REPORTS / "threshold_evaluation.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation completed.")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

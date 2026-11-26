
import time


class MockLLM:
    """
    Simulated LLM client for demonstrating semantic caching.

    Replace this class with a real provider integration when needed.
    """

    def __init__(self, simulated_latency: float = 0.5):
        if simulated_latency < 0:
            raise ValueError("Latency cannot be negative.")

        self.simulated_latency = simulated_latency
        self.call_count = 0

    def generate(self, question: str) -> str:
        """Generate a deterministic demonstration response."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        self.call_count += 1
        time.sleep(self.simulated_latency)

        text = question.lower()

        if "docker" in text or "container" in text:
            answer = (
                "Docker is a platform for building, packaging, "
                "and running applications in containers. "
                "Containers package applications with their dependencies "
                "to support consistent environments."
            )
        elif "kubernetes" in text or "k8s" in text:
            answer = (
                "Kubernetes is a container orchestration platform. "
                "It automates the deployment, scaling, and management "
                "of containerized applications."
            )
        elif "terraform" in text:
            answer = (
                "Terraform is an infrastructure-as-code tool. "
                "It allows engineers to define, provision, and manage "
                "infrastructure using configuration files."
            )
        elif "aws" in text or "cloud" in text:
            answer = (
                "Cloud computing provides on-demand access to computing "
                "resources such as servers, storage, databases, and "
                "networking over the internet."
            )
        elif "python" in text:
            answer = (
                "Python is a high-level programming language widely used "
                "for automation, application development, data analysis, "
                "and machine learning."
            )
        else:
            answer = (
                f"This is a simulated LLM response to: {question.strip()}. "
                "A production LLM would generate a context-aware answer."
            )

        return answer

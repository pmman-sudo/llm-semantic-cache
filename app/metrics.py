
class CacheMetrics:
    """Track cache performance and estimated cost savings."""

    def __init__(self, cost_per_llm_call: float = 0.002):
        if cost_per_llm_call < 0:
            raise ValueError("Cost per call cannot be negative.")

        self.cost_per_llm_call = cost_per_llm_call
        self.hits = 0
        self.misses = 0
        self.total_latency = 0.0
        self.latencies = []

    def record(self, cache_hit: bool, latency: float):
        """Record the result and latency of one request."""
        if latency < 0:
            raise ValueError("Latency cannot be negative.")

        if cache_hit:
            self.hits += 1
        else:
            self.misses += 1

        self.total_latency += latency
        self.latencies.append(latency)

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0

        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0

        return self.misses / self.total_requests

    @property
    def average_latency(self) -> float:
        if self.total_requests == 0:
            return 0.0

        return self.total_latency / self.total_requests

    @property
    def estimated_cost_saved(self) -> float:
        return self.hits * self.cost_per_llm_call

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": round(self.hit_rate * 100, 2),
            "miss_rate": round(self.miss_rate * 100, 2),
            "average_latency_seconds": round(
                self.average_latency, 4
            ),
            "estimated_cost_saved": round(
                self.estimated_cost_saved, 6
            ),
        }

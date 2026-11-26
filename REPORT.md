# Technical Project Report: LLM Semantic Cache



**Author:** Iyen Paul  

**Project:** LLM Semantic Cache  

**Programming Language:** Python  

**Date:** November 2026



---
## 1. Introduction



Large Language Model (LLM) applications may receive repeated questions expressed in different ways. Without an appropriate caching mechanism, each request can trigger another model inference, increasing response latency and computational or API costs.



This project implements a semantic caching system that identifies questions with similar meanings and retrieves previously stored responses instead of generating new ones.



The solution combines ONNX-based text embeddings, FAISS vector similarity search, and SQLite persistence to demonstrate an approach to reducing redundant LLM requests.



## 2. Project Objectives



The primary objectives were to:



1\. Implement semantic text embeddings for user questions.

2\. Use FAISS to identify semantically similar requests.

3\. Store questions, responses and embeddings persistently in SQLite.

4\. Retrieve cached responses when similarity exceeds a configured threshold.

5\. Track cache hits, misses, request latency and estimated cost savings.

6\. Evaluate performance against a baseline without semantic caching.

7\. Validate the implementation through automated tests.



## 3. System Architecture



The system follows a request-processing workflow.



1\. A user submits a question.

2\. The embedding model converts the question into a numerical vector.

3\. FAISS searches for similar vectors in the cache index.

4\. The similarity score is compared against a configured threshold.

5\. If a suitable match is found, the stored response is returned.

6\. If no suitable match exists, the request is passed to the response generator.

7\. The generated response and its embedding are stored in SQLite.

8\. Cache metrics are updated.



The FAISS index is reconstructed from persistent SQLite records when the cache is initialized.



## 4. Technologies Used



| Technology | Application |

|---|---|

| Python | Application development |

| ONNX Runtime | Embedding inference |

| GPTCache ONNX integration | Embedding model interface |

| FAISS | Vector indexing and similarity search |

| SQLite | Persistent cache storage |

| NumPy | Vector processing |

| Matplotlib | Benchmark visualization |

| Pytest | Automated testing |



## 5. Implementation



### 5.1 Embedding Generation



The embedding component uses GPTCache's ONNX integration to convert text into 768-dimensional vectors.



The vectors are normalized before similarity comparison. This allows the system to use FAISS inner-product search as cosine similarity search.



### 5.2 Semantic Cache



The cache component uses FAISS `IndexFlatIP` to retrieve the closest matching vector.



A configurable similarity threshold of 0.55 determines whether a result is considered a cache hit.



SQLite stores the original question, generated response, embedding data and creation timestamp. Stored embeddings are loaded into the FAISS index when the application starts.



### 5.3 Response Generation



A simulated response generator called `MockLLM` is used to demonstrate the application without requiring a real LLM provider or API credentials.



It generates deterministic example responses and supports configurable simulated latency.



This makes it possible to evaluate cache behavior independently of external API availability.



### 5.4 Metrics



The metrics component tracks:



\- Total cache hits.

\- Total cache misses.

\- Cache hit rate.

\- Average request latency.

\- Estimated cost savings.



Cost savings are calculated using an assumed cost per LLM call of $0.002. This is an illustrative benchmark parameter, not an actual provider price.



## 6. Testing and Validation



The initial automated test suite contained eight tests.



| Test | Purpose |

|---|---|

| Cache storage and retrieval | Verify stored responses can be retrieved |

| Similar-question matching | Confirm semantically related questions can hit |

| Unrelated-question rejection | Confirm unrelated questions are rejected |

| Persistence | Verify records survive cache restarts |

| Invalid embedding dimension | Validate vector dimensions |

| Zero-vector validation | Reject invalid zero vectors |

| Empty-question validation | Reject empty questions |

| Record counting | Verify cache record counts |



**Initial automated test result: 8 passed.**



Manual integration tests also confirmed semantic retrieval and persistence after closing and reopening the SQLite database.



## 7. Benchmark Methodology



The benchmark compared two execution modes:



**Baseline:** Every question is passed directly to the simulated LLM.



**Semantic cache:** Each question is embedded and checked against the cache. Only cache misses trigger the simulated LLM.



The benchmark configuration was:



| Parameter | Value |

|---|---:|

| Questions per trial | 15 |

| Number of trials | 5 |

| Total requests measured | 75 |

| Simulated LLM latency | 1 second |

| Similarity threshold | 0.55 |

| Assumed cost per LLM call | $0.002 |

| Cache state | Fresh for each trial |



The embedding model was warmed up before measuring cached requests. Each trial used a separate temporary SQLite database to ensure that it started with an empty cache.



The benchmark measured request-processing latency, including embedding generation, cache lookup and storage operations for the semantic-cache implementation.



## 8. Benchmark Results



The average results across five trials were:



| Metric | Baseline | Semantic Cache |

|---|---:|---:|

| Average total latency | 15.0144 s | 9.7736 s |

| Median total latency | 15.0077 s | 9.7000 s |

| Average latency per request | 1.0010 s | 0.6516 s |

| LLM calls per trial | 15 | 6 |



Additional semantic-cache results:



\- Average cache hits: 9 per trial.

\- Average cache misses: 6 per trial.

\- Cache hit rate: 60%.

\- Average LLM calls avoided: 9 per trial.

\- Average time saved: 5.2407 seconds per trial.

\- Average latency improvement: 34.9%.

\- Estimated cost savings: $0.018 per trial.



The latency improvement was calculated as:



\\\[

\\text{Improvement} =

\\frac{\\text{Baseline latency} - \\text{Cached latency}}

{\\text{Baseline latency}} \\times 100

\\]



The measured improvement was approximately 34.9%.



## 9. Analysis and Discussion



The results indicate that the semantic cache reduced the number of simulated LLM calls from 15 to 6 per trial, representing a 60% reduction.



The cached implementation also achieved a lower average total execution time. It processed the workload in approximately 9.77 seconds compared with 15.01 seconds for the baseline.



However, an earlier experiment using a simulated LLM latency of 0.1 seconds produced a different outcome. In that experiment, embedding generation and cache retrieval overhead exceeded the time saved by avoiding model calls.



This demonstrates that the effectiveness of semantic caching depends on workload characteristics, embedding overhead, cache hit rate and the latency of the underlying LLM service.



The benchmark results are encouraging, but they should be interpreted as controlled prototype measurements rather than production performance guarantees.



## 10. Challenges Encountered



### Dependency compatibility



The initial attempt to install the sentence-transformers stack encountered download timeouts and dependency compatibility issues.



The implementation was adapted to use GPTCache's ONNX embedding integration. Transformers and Hugging Face Hub versions were also adjusted to maintain compatibility.



### Embedding inference



The embedding workflow required configuring and validating the ONNX model and ensuring that generated vectors had the expected dimensions.



### Semantic similarity



A similarity threshold was needed to distinguish related questions from unrelated ones. Manual tests were used to verify that similar Docker questions could match while unrelated cooking questions were rejected.



### Persistent storage



The FAISS index needed to remain consistent with the records stored in SQLite. The implementation therefore reconstructs the vector index from persisted records when the cache starts.



### Performance measurement



The first benchmark revealed that semantic caching can introduce additional processing overhead. This motivated a second experiment with a longer simulated LLM delay, followed by five repeated trials and embedding-model warm-up.



## 11. Limitations



The current implementation has several limitations:



1\. The response generator is simulated rather than connected to a real LLM API.

2\. The benchmark uses only 15 synthetic questions per trial.

3\. The assumed API cost does not represent actual provider billing.

4\. The benchmark does not evaluate concurrent workloads or distributed deployments.

5\. The selected similarity threshold may require tuning for different applications.

6\. Production reliability, security and scalability have not yet been evaluated.



## 12. Future Improvements



Possible future enhancements include:



\- Integration with a real LLM provider.

\- Configurable cache expiration and invalidation.

\- Improved cache consistency and concurrency handling.

\- Larger benchmark datasets and more diverse workloads.

\- Evaluation with real model response times and provider pricing.

\- Docker containerization.

\- Cloud deployment.

\- Monitoring dashboards and operational metrics.

\- Authentication and API access controls.



## 13. Conclusion



This project demonstrates the design and implementation of a semantic caching system for LLM applications.



By combining ONNX-based embeddings, FAISS similarity search and SQLite persistence, the system can recognize semantically similar questions and reuse stored responses.



In the five-trial benchmark, the implementation achieved a 60% cache hit rate and a 34.9% average latency improvement under a one-second simulated LLM delay.



The project also demonstrated that caching overhead must be considered when evaluating performance, particularly when the underlying model responds quickly.



Overall, the implementation provided practical experience in Python application development, vector similarity search, persistent storage, automated testing, performance measurement and technical documentation.



---
**Author:** Iyen Paul  

**Project:** LLM Semantic Cache  

**Repository:** To be added after GitHub publication.


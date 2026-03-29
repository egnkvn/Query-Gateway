## Training Classifier

I use the `all-MiniLM-L6-v2` sentence embedding model together with Logistic Regression to train a classifier that distinguishes between fast-path and slow-path queries. The training data is based on `databricks/databricks-dolly-15k` filtered by category. (fast: classification, summarization; slow: creative_writing, general_qa)

```bash
python train.py
```
The classifier is saved as `classifier.joblib` after training.

## Docker
The classifier weight is already in the image. To run the server locally with Docker:
```bash
docker build -t cycarrier .
docker run -d -p 8000:8000 --name cycarrier
```
Then send a test request:
```bash
curl -s -D - -X POST http://localhost:8000/v1/query-process \
     -H "Content-Type: application/json" \
     -d '{"text": "Summarize this article."}' 
```
Example response:
```
HTTP/1.1 200 OK
x-router-latency: 123

{"label": "0"}
```

## System Design
This system is an async semantic routing gateway built with FastAPI. Its purpose is to classify each incoming query into one of two paths:

- **Fast Path (Label 0)**: `classification`, `summarization`
- **Slow Path (Label 1)**: `creative_writing`, `general_qa`

The router uses the `all-MiniLM-L6-v2` sentence embedding model together with a Logistic Regression classifier.  

### Overall Flow
1. Receive an HTTP request
2. Check the cache to see whether the query has already been classified and is still within its TTL
3. If cache hit, return the label directly
4. If cache miss, add the query to the batch queue
5. Wait until either the batching window expires or the maximum batch size is reached
6. Trigger the batch worker to run inference on all queued requests in a batch
7. Write the predicted result back to the corresponding request
8. Update the cache with the predicted result and a new expiration time
9. Route the request to the corresponding Fast and Slow path simulation

### Classification Model

* Embedding Model: `all-MiniLM-L6-v2`
* Classifier: `Logistic Regression`

### Asynchronous Design and High Concurrency

The server use FasAPI and async def to implement the asynchronous endpoint, and the routing pipeline is also asynchronous. 

* Incoming requests would be placed to working queue, and the batch worker would process the pending requests in queue in the background.
* The classifer also use `asyncio.to_thread` to run the inference in a separate thread to avoid blocking the event loop.

### Batch Inference

To support batch inference, each HTTP request is first placed into a working queue. A background batch worker then collects pending requests from the queue and processes them together, either when the batching window expires or when the maximum batch size is reached.

* **Batching window**: `20 ms`
* **Maximum batch size**: `32`

The batching window is designed to improve throughput by aggregating nearby requests into one inference call. The maximum batch size limits the cost of each batch and prevents classifier overhead, which could increase latency.

### Cache

The router uses an LRU-style cache with TTL, implemented with `OrderedDict`, to store recently seen queries together with their expiration times. First, each query is normalized using `strip()` and `lower()`. The router then checks whether the normalized query exists in the cache. If there is a cache hit, the request is routed directly to its corresponding path, and the expiration time is refreshed. If there is a cache miss, the request is added to the working queue and processed by the batch worker.

* **Cache TTL**: `300 ms`
* **Cache max size**: `1000`

The TTL prevents old results from staying in the cache permanently. This is useful because if the classifier is updated, previously cached results should eventually expire. The cache size limit is used to bound memory usage and prevent unbounded cache growth.

## Router latency and Throughput

```
###############################################################
Mixed traffic: 75 Fast | 45 Slow | 30 Cache-hit
###############################################################
Total requests        : 150
Successful            : 150
Failed                : 0

Wall-clock Time       : 3283.39 ms
Throughput            : 45.68 req/s

  ┌──────────────────────────────────────────────────────────┐
  │                Router Latency Summary (ms)               │
  ├────────────┬──────────┬──────────┬──────────┬────────────┤
  │  Metric    │   Avg    │   P50    │   P95    │    P99     │
  ├────────────┼──────────┼──────────┼──────────┼────────────┤
  │  Router    │   778.39 │   248.00 │  2838.00 │    3111.00 │
  │  Client E2E│   859.01 │   348.28 │  2909.94 │    3207.54 │
  └────────────┴──────────┴──────────┴──────────┴────────────┘

  Label Distribution  :
    Fast Path (0)       105 (70.0%)
    Slow Path (1)        45 (30.0%)
```

## Classifier Results

Use 10% of `databricks-dolly-15k` filtered by category. (fast: classification, summarization; slow: creative_writing, general_qa) as test set (623 samples), the results are as follows:

**Overall Accuracy**: 0.88

| Path | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Fast Path (0)** | 0.91 | 0.87 | 0.89 | 338 |
| **Slow Path (1)** | 0.85 | 0.89 | 0.87 | 285 |








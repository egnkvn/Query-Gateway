import asyncio
import time
import uuid
import statistics
import aiohttp
from collections import Counter, defaultdict
from math import ceil

API_URL = "http://127.0.0.1:8000/v1/query-process"
REQUEST_TIMEOUT = 15
BATCH_SIZE_LIMIT = 32
BATCH_TEST_CONCURRENCY = 100

# 若 server 的 TTL 很大（例如 300000 ms = 5 分鐘），TTL 測試會等很久
CACHE_TTL_MS = 3000         # 建議測試時把 server 也設成 3000 ms
CACHE_MAX_SIZE = 20         # 建議測試時把 server 也設成 20，比較快驗證 eviction

FAST_TEMPLATE = "Summarize this report in two bullets. case={id}"
SLOW_TEMPLATE = "Write a short fantasy story about a dragon. case={id}"

# 可選 debug headers；若 server 沒有回，腳本會自動退回 latency heuristic
DEBUG_CACHE_HEADER = "x-cache-hit"    # "1" / "0"
DEBUG_BATCH_SIZE_HEADER = "x-batch-size"
DEBUG_BATCH_ID_HEADER = "x-batch-id"


def mean_or_zero(xs):
    return statistics.mean(xs) if xs else 0.0


def get_percentile(xs, p):
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    idx = int(len(sorted_xs) * (p / 100))
    idx = min(len(sorted_xs) - 1, idx)
    return sorted_xs[idx]


def p50(xs):
    return get_percentile(xs, 50)


def p95(xs):
    return get_percentile(xs, 95)


def p99(xs):
    return get_percentile(xs, 99)


def parse_optional_bool(v):
    if v is None:
        return None
    return v in ("1", "true", "True", "YES", "yes")


def parse_optional_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except:
        return None


async def single_request(session, query, start_event=None):
    if start_event is not None:
        await start_event.wait()

    payload = {"text": query}
    t0 = time.perf_counter()

    try:
        async with session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT) as resp:
            body = await resp.json()
            t1 = time.perf_counter()

            return {
                "success": resp.status == 200,
                "status": resp.status,
                "query": query,
                "label": body.get("label"),
                "latency_ms": (t1 - t0) * 1000,
                "server_reported_ms": float(resp.headers.get("x-router-latency", 0)),
                "cache_hit": parse_optional_bool(resp.headers.get(DEBUG_CACHE_HEADER)),
                "batch_size": parse_optional_int(resp.headers.get(DEBUG_BATCH_SIZE_HEADER)),
                "batch_id": resp.headers.get(DEBUG_BATCH_ID_HEADER),
            }
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": str(e),
        }


async def burst_requests(session, queries):
    start_event = asyncio.Event()
    tasks = [asyncio.create_task(single_request(session, q, start_event)) for q in queries]

    await asyncio.sleep(0)  # 確保 task 都已建立
    wall_t0 = time.perf_counter()
    start_event.set()
    results = await asyncio.gather(*tasks)
    wall_t1 = time.perf_counter()

    return results, (wall_t1 - wall_t0) * 1000


def summarize_results(title, results, wall_time_ms=None):
    ok = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Total                 : {len(results)}")
    print(f"Successful            : {len(ok)}")
    print(f"Failed                : {len(failed)}")

    if failed:
        print("Sample Error          :", failed[0].get("error"))

    if not ok:
        return

    e2e = [r["latency_ms"] for r in ok]
    server = [r["server_reported_ms"] for r in ok]
    labels = Counter(r["label"] for r in ok)

    if wall_time_ms is not None:
        print(f"Wall-clock Time       : {wall_time_ms:.2f} ms")
        print(f"Throughput            : {len(ok)/(wall_time_ms/1000):.2f} req/s")

    print(f"Client E2E (ms)       : Avg: {mean_or_zero(e2e):.2f} | P50: {p50(e2e):.2f} | P95: {p95(e2e):.2f} | P99: {p99(e2e):.2f}")
    print(f"Router Latency (ms)   : Avg: {mean_or_zero(server):.2f} | P50: {p50(server):.2f} | P95: {p95(server):.2f} | P99: {p99(server):.2f}")

    print("Label Distribution    :")
    for lbl, cnt in sorted(labels.items()):
        name = "Fast Path (0)" if lbl == "0" else "Slow Path (1)"
        print(f"  {name:<18} {cnt}")

    cache_hits = [r["cache_hit"] for r in ok if r["cache_hit"] is not None]
    if cache_hits:
        print(f"Debug Cache Hits      : {sum(cache_hits)}/{len(cache_hits)}")

    batch_sizes = [r["batch_size"] for r in ok if r["batch_size"] is not None]
    if batch_sizes:
        print(f"Observed Batch Sizes  : {Counter(batch_sizes)}")


async def test_batch_inference(session, template, name, concurrency):
    queries = [template.format(id=f"{name}-{i}-{uuid.uuid4().hex[:8]}") for i in range(concurrency)]
    results, wall_time_ms = await burst_requests(session, queries)

    summarize_results(f"TEST 1 - BATCH INFERENCE ({name})", results, wall_time_ms)

    ok = [r for r in results if r["success"]]
    if not ok:
        return

    total_server_ms = sum(r["server_reported_ms"] for r in ok)
    efficiency = total_server_ms / wall_time_ms if wall_time_ms > 0 else 0
    theoretical_min_batches = ceil(concurrency / BATCH_SIZE_LIMIT)

    print(f"\nExpected min batches  : {theoretical_min_batches} (if batching works)")
    print(f"Sum server latencies  : {total_server_ms:.2f} ms")
    print(f"Efficiency gain       : {efficiency:.2f}x")

    batch_sizes = [r["batch_size"] for r in ok if r["batch_size"] is not None]
    if batch_sizes:
        print("Conclusion            : Deterministic via x-batch-size header")
    else:
        print("Conclusion            : Inferred via wall time / summed latency")
        if efficiency > 3:
            print("  ✅ Strong sign of concurrent batched processing")
        else:
            print("  ⚠️  Weak batching signal; verify server logs or add debug headers")


async def test_cache_basic(session):
    q = FAST_TEMPLATE.format(id=f"cache-basic-{uuid.uuid4().hex[:8]}")

    cold = await single_request(session, q)
    hot = [await single_request(session, q) for _ in range(5)]

    summarize_results("TEST 2 - CACHE HIT (cold once, then hot repeats)", [cold] + hot)

    hot_ok = [r for r in hot if r["success"]]
    if not hot_ok or not cold["success"]:
        return

    if hot_ok[0]["cache_hit"] is not None:
        print("\nConclusion            : Deterministic via x-cache-hit header")
        print(f"  First request cache hit?  {cold['cache_hit']}")
        print(f"  Later requests hit?       {[r['cache_hit'] for r in hot_ok]}")
    else:
        print("\nConclusion            : Heuristic only (random sleep adds noise)")
        print(f"  Cold server latency  : {cold['server_reported_ms']:.2f} ms")
        print(f"  Hot P50 latency      : {p50([r['server_reported_ms'] for r in hot_ok]):.2f} ms")
        print("  Expect hot requests to be somewhat faster on average")


async def test_cache_ttl(session):
    if CACHE_TTL_MS > 15000:
        print(f"\n[SKIP] TTL test skipped because CACHE_TTL_MS={CACHE_TTL_MS} is too large for quick testing.")
        print("       Set server/script TTL to ~3000ms for practical verification.")
        return

    q = FAST_TEMPLATE.format(id=f"cache-ttl-{uuid.uuid4().hex[:8]}")

    cold = await single_request(session, q)
    hit = await single_request(session, q)

    print(f"\nWaiting for TTL to expire: {CACHE_TTL_MS} ms ...")
    await asyncio.sleep((CACHE_TTL_MS / 1000) + 0.5)

    after_ttl = await single_request(session, q)

    summarize_results("TEST 3 - CACHE TTL EXPIRATION", [cold, hit, after_ttl])

    if hit["cache_hit"] is not None and after_ttl["cache_hit"] is not None:
        print("\nConclusion            : Deterministic via x-cache-hit header")
        print(f"  cold hit?            {cold['cache_hit']}")
        print(f"  immediate hit?       {hit['cache_hit']}")
        print(f"  after TTL hit?       {after_ttl['cache_hit']}")
    else:
        print("\nConclusion            : Heuristic only")
        print(f"  immediate hit server latency : {hit['server_reported_ms']:.2f} ms")
        print(f"  after TTL server latency     : {after_ttl['server_reported_ms']:.2f} ms")
        print("  After TTL it should look more like a cold request again")


async def test_cache_eviction(session):
    if CACHE_MAX_SIZE > 100:
        print(f"\n[SKIP] Eviction test skipped because CACHE_MAX_SIZE={CACHE_MAX_SIZE} is too large for quick testing.")
        print("       Set server/script cache size to ~20 for practical verification.")
        return

    queries = [
        FAST_TEMPLATE.format(id=f"evict-{i}-{uuid.uuid4().hex[:8]}")
        for i in range(CACHE_MAX_SIZE + 2)
    ]

    print(f"\nPriming cache with {len(queries)} unique queries ...")
    for q in queries:
        await single_request(session, q)

    oldest_again = await single_request(session, queries[0])
    newest_again = await single_request(session, queries[-1])

    summarize_results("TEST 4 - CACHE SIZE EVICTION", [oldest_again, newest_again])

    if oldest_again["cache_hit"] is not None and newest_again["cache_hit"] is not None:
        print("\nConclusion            : Deterministic via x-cache-hit header")
        print(f"  oldest query hit?    {oldest_again['cache_hit']}   (expected: False)")
        print(f"  newest query hit?    {newest_again['cache_hit']}   (expected: True)")
    else:
        print("\nConclusion            : Heuristic only")
        print(f"  oldest latency       : {oldest_again['server_reported_ms']:.2f} ms")
        print(f"  newest latency       : {newest_again['server_reported_ms']:.2f} ms")
        print("  Oldest should behave more like a cache miss; newest more like a hit")


OVERALL_CONCURRENCY = 150   # 同時送出的總 request 數
OVERALL_FAST_RATIO = 0.5    # 50% Fast Path queries
OVERALL_SLOW_RATIO = 0.3    # 30% Slow Path queries
# 剩下 20% 是重複 query (模擬 cache hit)


async def test_overall(session):
    """TEST 5 - OVERALL COMPREHENSIVE
    模擬真實混合流量：Fast + Slow + Cache-hit queries 同時並發，
    最後彙整出整體 Router Latency (P50/P95/P99) 與 Throughput。
    """
    n_fast  = int(OVERALL_CONCURRENCY * OVERALL_FAST_RATIO)
    n_slow  = int(OVERALL_CONCURRENCY * OVERALL_SLOW_RATIO)
    n_cache = OVERALL_CONCURRENCY - n_fast - n_slow  # 重複送以觸發 cache hit

    # 先送一個 warm-up 讓 cache 存好
    cache_query = FAST_TEMPLATE.format(id=f"overall-warm-{uuid.uuid4().hex[:8]}")
    await single_request(session, cache_query)

    queries = (
        [FAST_TEMPLATE.format(id=f"ovr-fast-{i}-{uuid.uuid4().hex[:8]}") for i in range(n_fast)]
        + [SLOW_TEMPLATE.format(id=f"ovr-slow-{i}-{uuid.uuid4().hex[:8]}") for i in range(n_slow)]
        + [cache_query] * n_cache          # 重複 query → 觸發 cache hit
    )

    results, wall_time_ms = await burst_requests(session, queries)

    ok     = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    server_latencies = [r["server_reported_ms"] for r in ok]
    e2e_latencies    = [r["latency_ms"]         for r in ok]
    throughput       = len(ok) / (wall_time_ms / 1000) if wall_time_ms > 0 else 0

    print(f"\n{'#'*70}")
    print(f"  TEST 5 - OVERALL COMPREHENSIVE BENCHMARK")
    print(f"  Mixed traffic: {n_fast} Fast | {n_slow} Slow | {n_cache} Cache-hit")
    print(f"{'#'*70}")
    print(f"Total requests        : {len(results)}")
    print(f"Successful            : {len(ok)}")
    print(f"Failed                : {len(failed)}")
    if failed:
        print("Sample Error          :", failed[0].get("error"))
    print(f"")
    print(f"Wall-clock Time       : {wall_time_ms:.2f} ms")
    print(f"Throughput            : {throughput:.2f} req/s")
    print(f"")
    print(f"  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │                Router Latency Summary (ms)               │")
    print(f"  ├────────────┬──────────┬──────────┬──────────┬────────────┤")
    print(f"  │  Metric    │   Avg    │   P50    │   P95    │    P99     │")
    print(f"  ├────────────┼──────────┼──────────┼──────────┼────────────┤")
    print(f"  │  Router    │ {mean_or_zero(server_latencies):>8.2f} │ {p50(server_latencies):>8.2f} │ {p95(server_latencies):>8.2f} │ {p99(server_latencies):>10.2f} │")
    print(f"  │  Client E2E│ {mean_or_zero(e2e_latencies):>8.2f} │ {p50(e2e_latencies):>8.2f} │ {p95(e2e_latencies):>8.2f} │ {p99(e2e_latencies):>10.2f} │")
    print(f"  └────────────┴──────────┴──────────┴──────────┴────────────┘")
    print(f"")

    labels = Counter(r["label"] for r in ok)
    print("  Label Distribution  :")
    for lbl, cnt in sorted(labels.items()):
        name = "Fast Path (0)" if lbl == "0" else "Slow Path (1)"
        print(f"    {name:<18} {cnt:>4} ({cnt/len(ok)*100:.1f}%)")
    print(f"{'#'*70}")


async def main():
    print(f"🚀 Testing server: {API_URL}")
    print(f"Batch size limit       : {BATCH_SIZE_LIMIT}")
    print(f"Batch concurrency      : {BATCH_TEST_CONCURRENCY}")
    print(f"Cache TTL (ms)         : {CACHE_TTL_MS}")
    print(f"Cache max size         : {CACHE_MAX_SIZE}")

    async with aiohttp.ClientSession() as session:
        await test_batch_inference(session, FAST_TEMPLATE, "fast", BATCH_TEST_CONCURRENCY)
        await test_batch_inference(session, SLOW_TEMPLATE, "slow", BATCH_TEST_CONCURRENCY)
        await test_cache_basic(session)
        await test_cache_ttl(session)
        await test_cache_eviction(session)
        await test_overall(session)


if __name__ == "__main__":
    asyncio.run(main())
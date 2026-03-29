import time
import random
import joblib
import asyncio
from typing import Optional
from collections import OrderedDict
from fastapi import FastAPI, Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

class SemanticRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_window: int = 20, max_batch_size: int = 32, cache_ttl: int = 300000, cache_max_size: int = 1000):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = joblib.load('classifier.joblib')
        self.batch_window = batch_window / 1000
        self.max_batch_size = max_batch_size
        self.queue = asyncio.Queue()

        self.cache_ttl = cache_ttl / 1000
        self.cache_max_size = cache_max_size
        self.cache = OrderedDict() 

    def _normalize_query(self, query: str) -> str:
        return query.strip().lower()

    def _set_cache(self, key: str, label: int):
        expire_time = time.monotonic() + self.cache_ttl
        self.cache[key] = (label, expire_time)
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_max_size:
            self.cache.popitem(last=False)

    def _get_cache(self, key: str) -> Optional[int]:
        item = self.cache.get(key)
        if item is None:
            return None
        
        label, expire_time = item
        now_time = time.monotonic()

        if now_time >= expire_time:
            del self.cache[key]
            return None
        
        self.cache[key] = (label, now_time + self.cache_ttl)
        self.cache.move_to_end(key)
        return label
        

    def _classify_batch(self, queries: list[str]) -> list[int]:
        embeddings = self.encoder.encode(queries)
        labels = self.classifier.predict(embeddings)
        return [int(x) for x in labels]

    async def _batch_worker(self):
        while True:
            await asyncio.sleep(self.batch_window)

            batch = []
            while not self.queue.empty() and len(batch) < self.max_batch_size:
                batch.append(self.queue.get_nowait())

            if len(batch) == 0:
                continue
                
            queries, futures = zip(*batch)
            labels = await asyncio.to_thread(self._classify_batch, list(queries))
            for fut, label in zip(futures, labels):
                if not fut.done():
                    fut.set_result(label)

    async def route(self, query: str) -> int:
        key = self._normalize_query(query)

        label = self._get_cache(key)
        if label == None:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await self.queue.put((query, fut))
            label = await fut
            self._set_cache(key, label)
        
        if label == 0:
            await asyncio.sleep(random.uniform(0.01, 0.2))
        else:
            await asyncio.sleep(random.uniform(1.0, 3.0))
        return label



router = SemanticRouter(cache_ttl=300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(router._batch_worker())

class QueryRequest(BaseModel):
    text: str

@app.post("/v1/query-process")
async def process_query(request: QueryRequest, response: Response):
    start_time = time.time()
    label = await router.route(request.text)
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    response.headers["x-router-latency"] = str(latency_ms)
    return {"label": str(label)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

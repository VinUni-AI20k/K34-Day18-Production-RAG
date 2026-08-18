from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


import os
from pinecone import Pinecone, PineconeException

class CrossEncoderReranker:
    def __init__(self, model_name: str = "bge-reranker-v2-m3"):
        self.model_name = model_name
        self._pc = None

    def _get_client(self):
        if self._pc is None:
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is missing")
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 -> top-k."""
        if not documents:
            return []
            
        pc = self._get_client()
        docs_texts = [doc["text"] for doc in documents]
        
        try:
            res = pc.inference.rerank(
                model=self.model_name,
                query=query,
                documents=docs_texts,
                top_n=len(documents), # Get all, we will truncate to top_k later to be safe, or we can just pass top_k
                return_documents=True
            )
        except PineconeException as e:
            err_msg = str(e).replace(os.environ.get("PINECONE_API_KEY", "NOT_SET"), "****")
            raise RuntimeError(f"Pinecone API failed: {type(e).__name__} - {err_msg}")
        except Exception as e:
            err_msg = str(e).replace(os.environ.get("PINECONE_API_KEY", "NOT_SET"), "****")
            raise RuntimeError(f"Pinecone inference error: {type(e).__name__} - {err_msg}")
            
        items = res.data
        # Ensure it's sorted by score descending
        try:
            items.sort(key=lambda x: x["score"], reverse=True)
        except TypeError: # if object
            items.sort(key=lambda x: x.score, reverse=True)
        
        results = []
        for i, item in enumerate(items[:top_k]):
            if isinstance(item, dict):
                idx = item["index"]
                score = float(item["score"])
            else:
                idx = item.index
                score = float(item.score)
                
            original_doc = documents[idx]
            results.append(RerankResult(
                text=original_doc["text"],
                original_score=original_doc.get("score", 0.0),
                rerank_score=score,
                metadata=original_doc.get("metadata", {}),
                rank=i
            ))
            
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")

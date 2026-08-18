from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize
    segmented = word_tokenize(text, format="text")
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            text = chunk.get("text", "")
            seg = segment_vietnamese(text)
            self.corpus_tokens.append(seg.split())
            
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
            
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in scored_indices[:top_k]:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=score,
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
            
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            class RemoteEncoder:
                def encode(self, texts, show_progress_bar=False):
                    import os
                    import time
                    import numpy as np
                    from huggingface_hub import InferenceClient
                    
                    client = InferenceClient(
                        provider="hf-inference",
                        api_key=os.environ["HF_TOKEN"]
                    )
                    batch_size = 16
                    all_embeddings = []
                    
                    is_single_string = False
                    if isinstance(texts, str):
                        texts = [texts]
                        is_single_string = True
                        
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        for attempt in range(5):
                            try:
                                emb = client.feature_extraction(batch, model="BAAI/bge-m3")
                                all_embeddings.extend(emb)
                                break
                            except Exception as e:
                                if attempt == 4:
                                    raise Exception(f"Embedding API failed: {e}")
                                time.sleep(2)
                                
                    result = np.array(all_embeddings)
                    if result.ndim == 3 and result.shape[1] == 1:
                        # Sometimes feature extraction returns [batch_size, 1, hidden_size]
                        result = result.squeeze(axis=1)
                    
                    if result.shape[-1] != 1024:
                        raise ValueError(f"Expected embedding dimension 1024, but got {result.shape[-1]}")
                        
                    if is_single_string:
                        return result[0]
                    return result
            self._encoder = RemoteEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        
        texts = [c["text"] for c in chunks]
        if not texts:
            return
            
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
            points.append(PointStruct(id=i, vector=vector.tolist(), payload=payload))
            
        self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        query_vector = self._get_encoder().encode(query).tolist()
        
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )
        
        results = []
        for pt in response.points:
            results.append(SearchResult(
                text=pt.payload["text"],
                score=pt.score,
                metadata=pt.payload,
                method="dense"
            ))
            
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    final_results = []
    for item in sorted_items[:top_k]:
        orig_res = item["result"]
        final_results.append(SearchResult(
            text=orig_res.text,
            score=item["score"],
            metadata=orig_res.metadata,
            method="hybrid"
        ))
        
    return final_results


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")

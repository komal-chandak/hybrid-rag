import numpy as np
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

FINAL_MAX_THRESHOLD = 0.45
FINAL_AVG_THRESHOLD = 0.30

class RetrievalService:
    def __init__(self, index, model, bm25, reranker):
        self.index = index
        self.model = model
        self.reranker = reranker
        self.bm25 = bm25
        self.executor = ThreadPoolExecutor(max_workers=4)

    # ---------------- PARALLEL HELPERS ----------------

    def _get_embedding(self, query):
        return self.model.encode(
            "search_query: " + query,
            normalize_embeddings=True
        )

    def _get_bm25(self, query):
        return self.bm25.encode_queries(query)

    async def _prepare_query(self, query):
        loop = asyncio.get_event_loop()

        embedding_task = loop.run_in_executor(
            self.executor,
            self._get_embedding,
            query
        )

        bm25_task = loop.run_in_executor(
            self.executor,
            self._get_bm25,
            query
        )

        embedding, sparse_vector = await asyncio.gather(
            embedding_task,
            bm25_task
        )

        return embedding, sparse_vector

    # ---------------- FALLBACK LOGIC ----------------
    @staticmethod
    def _should_fallback(chunks):
        # NOTE:
        # Fallback decision is based on empirically derived thresholds from a limited test set.
        # These thresholds may not generalize across all queries and should not be treated as
        # fully reliable for production decision-making.
        # Currently used as a soft signal (optional) and NOT enforced unless explicitly enabled.
        
        final_scores = [c["final_score"] for c in chunks]
        rerank_scores = [c["rerank_score"] for c in chunks]

        avg_final = sum(final_scores) / len(final_scores)
        max_final = max(final_scores)
        positive_reranks = sum(1 for r in rerank_scores if r > 0)

        if max_final < FINAL_MAX_THRESHOLD:
            return True

        if avg_final < FINAL_AVG_THRESHOLD:
            return True

        if positive_reranks == 0:
            return True

        return False

    @staticmethod
    def _normalize(scores):
        scores = np.array(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _fuse_scores(self, chunks, alpha=0.5):
        retr_scores = [c["score"] for c in chunks]
        rerank_scores = [c["rerank_score"] for c in chunks]

        retr_norm = self._normalize(retr_scores)
        rerank_norm = self._normalize(rerank_scores)

        for i, c in enumerate(chunks):
            c["final_score"] = (
                alpha * rerank_norm[i] +
                (1 - alpha) * retr_norm[i]
            )

        return sorted(chunks, key=lambda x: x["final_score"], reverse=True)

    @staticmethod
    def _reorder_neighbors_locally(chunks):
        def get_id(c):
            return int(c["id"].split("_")[-1])

        # Sort by ID first for grouping
        sorted_by_id = sorted(chunks, key=get_id)

        groups = []
        current = [sorted_by_id[0]]

        for prev, curr in zip(sorted_by_id, sorted_by_id[1:]):
            if get_id(curr) - get_id(prev) == 1:
                current.append(curr)
            else:
                groups.append(current)
                current = [curr]

        groups.append(current)

        # rebuild order based on original ranking priority
        id_to_chunk = {c["id"]: c for c in chunks}

        final = []
        for group in groups:
            if len(group) > 1:
                # keep logical order
                group = sorted(group, key=get_id)
            final.extend([id_to_chunk[c["id"]] for c in group])

        return final
    
    # ---------------- MAIN RETRIEVAL ----------------
    
    async def retrieve(self, query, fallback_signal, top_k=20, DEBUG=False):
        should_fallback = False
        embedding, sparse_vector = await self._prepare_query(query)
        # embedding = self.model.encode(
        #     "search_query: " + query,
        #     normalize_embeddings=True
        # )

        # sparse_vector = self.bm25.encode_queries(query) 
        results = self.index.query(
            vector=embedding.tolist(),
            sparse_vector=sparse_vector,
            top_k=top_k,
            include_metadata=True,
            alpha=0.7
        )
        matches = results["matches"]

        pairs = [
            (query, m["metadata"].get("content", ""))
            for m in matches
        ]
        scores = self.reranker.predict(pairs, batch_size = 8)
        for m, score in zip(matches, scores):
            m["rerank_score"] = float(score)


        final_chunks = self._fuse_scores(matches)
        final_chunks_top_3 = self._reorder_neighbors_locally(final_chunks[:3])
        if fallback_signal:
            should_fallback = self._should_fallback(final_chunks_top_3)

        if DEBUG:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"rag_debug_fallback/rag_debug_{ts}.json"

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, indent=2, default=str)

        return final_chunks_top_3, should_fallback  # return top 3 after rerank and retrieval

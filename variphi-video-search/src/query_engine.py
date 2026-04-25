"""
Query Engine
Encodes natural language queries, runs ANN retrieval,
applies temporal filters, and re-ranks results using CLIP similarity.
"""

import time
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import clip

from .indexer import VideoIndexer, FrameRecord, seconds_to_hms

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    rank: int
    timestamp: str          # HH:MM:SS
    timestamp_sec: float
    score: float            # cosine similarity (0-1)
    video_path: str
    frame_path: str
    query: str


def parse_temporal_filter(query: str) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Extract temporal constraints from query text.
    e.g. "person walking after 6pm" -> ("person walking", 64800.0, None)
    e.g. "between 18:00 and 20:00" -> ("", 64800.0, 72000.0)
    Returns (cleaned_query, start_sec, end_sec)
    """
    start_sec = None
    end_sec = None
    cleaned = query

    # Pattern: "between HH:MM and HH:MM"
    between_match = re.search(
        r"between\s+(\d{1,2}):(\d{2})\s+and\s+(\d{1,2}):(\d{2})", query, re.IGNORECASE
    )
    if between_match:
        sh, sm, eh, em = between_match.groups()
        start_sec = int(sh) * 3600 + int(sm) * 60
        end_sec = int(eh) * 3600 + int(em) * 60
        cleaned = query[: between_match.start()].strip() + " " + query[between_match.end():].strip()

    # Pattern: "after HH:MM" or "after Npm / Xam"
    after_match = re.search(
        r"after\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", cleaned, re.IGNORECASE
    )
    if after_match and start_sec is None:
        h = int(after_match.group(1))
        m = int(after_match.group(2) or 0)
        meridiem = (after_match.group(3) or "").lower()
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        start_sec = h * 3600 + m * 60
        cleaned = cleaned[: after_match.start()].strip() + " " + cleaned[after_match.end():].strip()

    # Pattern: "before HH:MM"
    before_match = re.search(
        r"before\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", cleaned, re.IGNORECASE
    )
    if before_match and end_sec is None:
        h = int(before_match.group(1))
        m = int(before_match.group(2) or 0)
        meridiem = (before_match.group(3) or "").lower()
        if meridiem == "pm" and h != 12:
            h += 12
        elif meridiem == "am" and h == 12:
            h = 0
        end_sec = h * 3600 + m * 60
        cleaned = cleaned[: before_match.start()].strip() + " " + cleaned[before_match.end():].strip()

    # Normalize whitespace
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        cleaned = query   # fallback

    return cleaned, start_sec, end_sec


def decompose_query(query: str) -> List[str]:
    """
    Decompose complex queries into sub-queries for multi-aspect retrieval.
    e.g. "person near entrance carrying a bag" → ["person near entrance", "person carrying bag", "bag near entrance"]
    Uses simple conjunction/preposition splitting heuristics.
    """
    sub_queries = [query]   # always include original

    # Split on conjunctions
    parts = re.split(r"\band\b|\bwith\b|\bcarrying\b|\bnear\b|\bholding\b", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if len(p.strip()) > 3]
    if len(parts) > 1:
        sub_queries.extend(parts)

    return list(dict.fromkeys(sub_queries))   # deduplicate, preserve order


class QueryEngine:
    def __init__(self, indexer: VideoIndexer, results_log: str = "results.json"):
        self.indexer = indexer
        self.results_log = Path(results_log)
        self.device = indexer.device
        self.model = indexer.model
        self.all_results: List[Dict] = []

        if self.results_log.exists():
            with open(self.results_log) as f:
                try:
                    self.all_results = json.load(f)
                except Exception:
                    self.all_results = []

    def search(
        self,
        query: str,
        top_k: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        rerank: bool = True,
        video_filter: Optional[str] = None,
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Main search method.
        Returns (results, metadata) where metadata contains timing/debug info.
        """
        if self.indexer.faiss_index is None:
            self.indexer.load()
        if self.indexer.faiss_index is None or len(self.indexer.records) == 0:
            return [], {"error": "Index is empty. Please index some videos first."}

        t0 = time.time()

        # Parse temporal filter from query text
        cleaned_query, auto_start, auto_end = parse_temporal_filter(query)
        start_sec = start_time if start_time is not None else auto_start
        end_sec = end_time if end_time is not None else auto_end

        # Sub-query decomposition
        sub_queries = decompose_query(cleaned_query)
        logger.info(f"Sub-queries: {sub_queries}")

        # Aggregate scores across sub-queries
        score_map: Dict[int, float] = {}   # embedding_idx -> best score

        for sq in sub_queries:
            scores, indices = self._ann_search(sq, top_k=top_k * 3)
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                if idx not in score_map or score > score_map[idx]:
                    score_map[idx] = float(score)

        # Map indices to records
        candidates: List[Tuple[FrameRecord, float]] = []
        for emb_idx, score in score_map.items():
            if emb_idx < len(self.indexer.records):
                rec = self.indexer.records[emb_idx]
                candidates.append((rec, score))

        # Temporal filter
        if start_sec is not None:
            candidates = [(r, s) for r, s in candidates if r.timestamp >= start_sec]
        if end_sec is not None:
            candidates = [(r, s) for r, s in candidates if r.timestamp <= end_sec]

        # Video filter
        if video_filter:
            candidates = [(r, s) for r, s in candidates if video_filter in r.video_path]

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[: top_k * 2]   # over-fetch for re-ranker

        # Re-ranking step: compute exact CLIP similarity for top candidates
        if rerank and len(candidates) > 1:
            candidates = self._rerank(cleaned_query, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        query_latency = (time.time() - t0) * 1000   # ms

        results = []
        for rank, (rec, score) in enumerate(candidates, start=1):
            results.append(
                SearchResult(
                    rank=rank,
                    timestamp=rec.timestamp_str,
                    timestamp_sec=rec.timestamp,
                    score=round(score, 4),
                    video_path=rec.video_path,
                    frame_path=rec.frame_path,
                    query=query,
                )
            )

        meta = {
            "query": query,
            "cleaned_query": cleaned_query,
            "sub_queries": sub_queries,
            "temporal_filter": {"start_sec": start_sec, "end_sec": end_sec},
            "num_results": len(results),
            "query_latency_ms": round(query_latency, 2),
            "reranked": rerank,
        }

        # Persist results
        self._log_results(results, meta)
        logger.info(f"Query '{query}' → {len(results)} results in {query_latency:.1f}ms")

        return results, meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()

    def _ann_search(self, query_text: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_emb = self._encode_text(query_text)
        faiss_k = min(top_k, self.indexer.faiss_index.ntotal)
        scores, indices = self.indexer.faiss_index.search(query_emb, faiss_k)
        return scores[0], indices[0]

    def _rerank(
        self,
        query_text: str,
        candidates: List[Tuple[FrameRecord, float]],
        top_k: int,
    ) -> List[Tuple[FrameRecord, float]]:
        """Re-rank using exact CLIP text-image cosine similarity."""
        query_emb = self._encode_text(query_text)   # (1, 512)
        query_emb = query_emb / np.linalg.norm(query_emb)

        reranked = []
        for rec, _ in candidates:
            emb_idx = rec.embedding_idx
            if emb_idx < self.indexer.faiss_index.ntotal:
                # Reconstruct embedding from FAISS index
                stored_emb = np.zeros((1, 512), dtype=np.float32)
                self.indexer.faiss_index.reconstruct(int(emb_idx), stored_emb[0])
                sim = float(np.dot(query_emb, stored_emb.T))
                reranked.append((rec, sim))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _log_results(self, results: List[SearchResult], meta: Dict):
        entries = []
        for r in results:
            d = asdict(r)
            d["meta"] = meta
            entries.append(d)
        self.all_results.extend(entries)
        with open(self.results_log, "w") as f:
            json.dump(self.all_results, f, indent=2)

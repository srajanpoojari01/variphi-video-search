#!/usr/bin/env python3
"""
Evaluation protocol — Precision@K and MRR over labelled test queries.

Ground truth format (eval_data.json):
[
  {
    "query": "person near the entrance carrying a bag",
    "relevant_timestamps": [12.0, 34.5, 89.0],   // seconds, tolerance ±5s
    "video": "optional/path/filter.mp4"
  }
]
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def precision_at_k(retrieved: List[float], relevant: List[float], k: int, tol: float = 5.0) -> float:
    top_k = retrieved[:k]
    hits = sum(
        1 for ts in top_k
        if any(abs(ts - gt) <= tol for gt in relevant)
    )
    return hits / k if k > 0 else 0.0


def reciprocal_rank(retrieved: List[float], relevant: List[float], tol: float = 5.0) -> float:
    for rank, ts in enumerate(retrieved, start=1):
        if any(abs(ts - gt) <= tol for gt in relevant):
            return 1.0 / rank
    return 0.0


def evaluate(eval_data: List[Dict], index_dir: str, thumbnail_dir: str, top_k: int = 10, tol: float = 5.0):
    from src.indexer import VideoIndexer
    from src.query_engine import QueryEngine

    indexer = VideoIndexer(index_dir=index_dir, thumbnail_dir=thumbnail_dir)
    indexer.load()
    if not indexer.records:
        logger.error("Index is empty!")
        return

    engine = QueryEngine(indexer, results_log="eval_results.json")

    p_at_1_list, p_at_k_list, rr_list = [], [], []

    print(f"\n{'Query':<50} {'P@1':>6} {'P@K':>6} {'RR':>6}")
    print("─" * 72)

    for item in eval_data:
        query = item["query"]
        relevant = item.get("relevant_timestamps", [])
        vid_filter = item.get("video")

        results, meta = engine.search(query, top_k=top_k, video_filter=vid_filter, rerank=True)
        retrieved = [r.timestamp_sec for r in results]

        p1 = precision_at_k(retrieved, relevant, k=1, tol=tol)
        pk = precision_at_k(retrieved, relevant, k=top_k, tol=tol)
        rr = reciprocal_rank(retrieved, relevant, tol=tol)

        p_at_1_list.append(p1)
        p_at_k_list.append(pk)
        rr_list.append(rr)

        short_q = query[:48] + ".." if len(query) > 48 else query
        print(f"{short_q:<50} {p1:>6.2f} {pk:>6.2f} {rr:>6.2f}")

    n = len(eval_data)
    mean_p1 = sum(p_at_1_list) / n
    mean_pk = sum(p_at_k_list) / n
    mrr = sum(rr_list) / n

    print("─" * 72)
    print(f"{'MEAN':50} {mean_p1:>6.2f} {mean_pk:>6.2f} {mrr:>6.2f}")
    print()
    print(f"  MAP@{top_k}  : {mean_pk:.4f}")
    print(f"  MRR     : {mrr:.4f}")
    print(f"  P@1     : {mean_p1:.4f}")
    print()

    report = {
        "top_k": top_k,
        "tolerance_sec": tol,
        "num_queries": n,
        "mean_precision_at_1": round(mean_p1, 4),
        f"mean_precision_at_{top_k}": round(mean_pk, 4),
        "mrr": round(mrr, 4),
    }
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Evaluation report saved → eval_report.json")
    return report


DEFAULT_EVAL_DATA = [
    {
        "query": "person near the entrance",
        "relevant_timestamps": [],
        "note": "Fill in ground-truth timestamps after indexing your video"
    }
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=None, help="JSON file with ground-truth queries")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--thumbnail-dir", default="thumbnails")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=5.0, help="Timestamp tolerance in seconds")
    args = parser.parse_args()

    if args.eval_file and Path(args.eval_file).exists():
        with open(args.eval_file) as f:
            eval_data = json.load(f)
    else:
        logger.warning("No eval file provided. Using placeholder data.")
        eval_data = DEFAULT_EVAL_DATA

    evaluate(eval_data, args.index_dir, args.thumbnail_dir, args.top_k, args.tolerance)

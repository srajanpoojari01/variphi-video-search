#!/usr/bin/env python3
"""
Benchmark script — measures indexing throughput, query latency, and memory.
Creates a synthetic test video if none is provided.
"""

import argparse
import logging
import time
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_video(path: str, duration_sec: int = 60, fps: int = 25):
    """Create a test video using OpenCV without external dependencies."""
    import cv2
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    n = duration_sec * fps
    for i in range(n):
        # Animated frame: colour ramps + timestamp text
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * (i % fps) / fps)
        frame[:, :, 1] = int(255 * ((i // fps) % 60) / 60)
        frame[:, :, 2] = 128
        cv2.putText(frame, f"Frame {i:05d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        writer.write(frame)
    writer.release()
    logger.info(f"Synthetic video: {path} ({duration_sec}s, {fps}fps, {n} frames)")


def benchmark(video_path: str, queries: list, top_k: int = 10):
    from src.indexer import VideoIndexer
    from src.query_engine import QueryEngine

    tmp_dir = tempfile.mkdtemp()
    index_dir = os.path.join(tmp_dir, "index")
    thumb_dir = os.path.join(tmp_dir, "thumbs")
    results_log = os.path.join(tmp_dir, "results.json")

    # ── Indexing benchmark ──────────────────────────────────────────
    logger.info("=== Indexing Benchmark ===")
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1024**2

    indexer = VideoIndexer(index_dir=index_dir, thumbnail_dir=thumb_dir)
    t_start = time.perf_counter()
    stats = indexer.index_video(video_path)
    t_index = time.perf_counter() - t_start

    mem_after = proc.memory_info().rss / 1024**2
    peak_mem = max(mem_before, mem_after)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║             INDEXING BENCHMARK RESULTS           ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Frames indexed    : {stats['frames_indexed']:<28} ║")
    print(f"║  Total time        : {t_index:.2f}s{'':<24} ║")
    print(f"║  Throughput        : {stats['throughput_fps']:.1f} frames/sec{'':<18} ║")
    print(f"║  Peak memory       : {peak_mem:.0f} MB{'':<24} ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── Query latency benchmark ─────────────────────────────────────
    logger.info("=== Query Latency Benchmark ===")
    engine = QueryEngine(indexer, results_log=results_log)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        results, meta = engine.search(q, top_k=top_k, rerank=True)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        print(f"  Query: \"{q[:50]}\"")
        print(f"    → {meta['num_results']} results in {lat:.1f}ms")

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    min_lat = min(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0

    print("\n╔══════════════════════════════════════════════════╗")
    print("║             QUERY LATENCY RESULTS                ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Queries run       : {len(queries):<28} ║")
    print(f"║  Avg latency       : {avg_lat:.1f} ms{'':<23} ║")
    print(f"║  Min latency       : {min_lat:.1f} ms{'':<23} ║")
    print(f"║  Max latency       : {max_lat:.1f} ms{'':<23} ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # Save report
    report = {
        "indexing": {
            "frames_indexed": stats["frames_indexed"],
            "elapsed_sec": stats["elapsed_sec"],
            "throughput_frames_per_sec": stats["throughput_fps"],
            "peak_memory_mb": round(peak_mem, 1),
        },
        "query": {
            "num_queries": len(queries),
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "device": "CPU",
        },
    }
    report_path = "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to video file (auto-creates synthetic if omitted)")
    parser.add_argument("--duration", type=int, default=60, help="Synthetic video duration in seconds")
    args = parser.parse_args()

    if args.video and Path(args.video).exists():
        video = args.video
    else:
        logger.info("No video provided — generating synthetic test video …")
        video = "/tmp/bench_test.mp4"
        create_synthetic_video(video, duration_sec=args.duration)

    test_queries = [
        "person walking near the entrance",
        "red vehicle parked in zone 3",
        "two people talking near the server rack",
        "anything unusual happening in the corridor",
        "person carrying a bag",
    ]

    benchmark(video, test_queries)

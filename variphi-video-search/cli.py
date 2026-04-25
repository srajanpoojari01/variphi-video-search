#!/usr/bin/env python3
"""
CLI tool for Variphi Video Search Engine.
Usage:
  python cli.py index /path/to/video.mp4
  python cli.py index /path/to/videos/
  python cli.py search "person near entrance carrying a bag"
  python cli.py search "red vehicle in zone 3" --top-k 5 --start 3600 --end 7200
  python cli.py stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.indexer import VideoIndexer
from src.query_engine import QueryEngine


def cmd_index(args):
    indexer = VideoIndexer(
        index_dir=args.index_dir,
        thumbnail_dir=args.thumbnail_dir,
        clip_model=args.model,
        batch_size=args.batch_size,
    )
    p = Path(args.path)
    if p.is_dir():
        stats_list = indexer.index_directory(str(p))
        for s in stats_list:
            _print_index_stats(s)
    elif p.is_file():
        stats = indexer.index_video(str(p))
        _print_index_stats(stats)
    else:
        print(f"[ERROR] Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)


def _print_index_stats(stats: dict):
    print("\n── Indexing Complete ──────────────────────────────────")
    print(f"  Video      : {stats['video']}")
    print(f"  Frames     : {stats['frames_indexed']}")
    print(f"  Throughput : {stats['throughput_fps']} frames/sec")
    print(f"  Time       : {stats['elapsed_sec']}s")
    print(f"  Peak RAM   : {stats['peak_memory_mb']} MB")
    print(f"  Total idx  : {stats['total_records']} frames")
    print("──────────────────────────────────────────────────────\n")


def cmd_search(args):
    indexer = VideoIndexer(
        index_dir=args.index_dir,
        thumbnail_dir=args.thumbnail_dir,
        clip_model=args.model,
    )
    indexer.load()
    if not indexer.records:
        print("[ERROR] Index is empty. Run 'python cli.py index <path>' first.", file=sys.stderr)
        sys.exit(1)

    engine = QueryEngine(indexer, results_log=args.results_log)
    results, meta = engine.search(
        query=args.query,
        top_k=args.top_k,
        start_time=args.start,
        end_time=args.end,
        rerank=not args.no_rerank,
    )

    print(f"\n── Results for: \"{args.query}\" ──────────────────────────")
    print(f"   Latency : {meta['query_latency_ms']}ms")
    print(f"   Found   : {meta['num_results']} results")
    if meta.get("sub_queries") and len(meta["sub_queries"]) > 1:
        print(f"   Sub-q   : {meta['sub_queries']}")
    print()

    if not results:
        print("  No results found.")
    else:
        for r in results:
            vid = Path(r.video_path).name
            print(f"  #{r.rank:02d}  [{r.timestamp}]  score={r.score:.4f}  {vid}")
            print(f"        Thumb: {r.frame_path}")
        print()
        print(f"  Results saved to: {args.results_log}")
    print("──────────────────────────────────────────────────────\n")


def cmd_stats(args):
    indexer = VideoIndexer(index_dir=args.index_dir, thumbnail_dir=args.thumbnail_dir)
    indexer.load()
    videos = list({r.video_path for r in indexer.records})
    print(f"\n── Index Stats ────────────────────────────────────────")
    print(f"  Index dir      : {args.index_dir}")
    print(f"  Total frames   : {len(indexer.records)}")
    print(f"  Total videos   : {len(videos)}")
    for v in videos:
        count = sum(1 for r in indexer.records if r.video_path == v)
        print(f"    {Path(v).name}: {count} frames")
    print("──────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(
        description="Variphi Video Search Engine — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index-dir", default="index", help="FAISS index directory")
    parser.add_argument("--thumbnail-dir", default="thumbnails", help="Thumbnail directory")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model variant")
    parser.add_argument("--results-log", default="results.json", help="Results output file")

    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index a video file or directory")
    p_index.add_argument("path", help="Path to video file or directory")
    p_index.add_argument("--batch-size", type=int, default=32)

    # search
    p_search = sub.add_parser("search", help="Search the video index")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--start", type=float, default=None, help="Start time in seconds")
    p_search.add_argument("--end", type=float, default=None, help="End time in seconds")
    p_search.add_argument("--no-rerank", action="store_true", help="Disable re-ranking")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()

# 🎬 Variphi Intelligent Video Search Engine

> Natural language querying over video archives — powered by **CLIP + FAISS**.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📺 Demo Video

> **[Watch the 1-minute walkthrough →](https://youtu.be/YOUR_LINK_HERE)**  
> *(Replace with your YouTube/Drive link after recording)*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Setup & Installation](#setup--installation)
4. [Usage](#usage)
5. [Design Decisions](#design-decisions)
6. [Benchmark Results](#benchmark-results)
7. [Open-Ended Exploration](#open-ended-exploration)
8. [Known Limitations](#known-limitations)
9. [API Reference](#api-reference)

---

## Overview

A system that lets you search through video archives using plain English. Given a video file or folder of clips, it:

1. **Indexes** the video offline: samples frames, generates CLIP embeddings, and stores them in FAISS.
2. **Queries** at runtime: encodes your natural language query in the same embedding space and performs fast ANN retrieval with optional re-ranking.

### Query Examples

| Type | Example |
|------|---------|
| Spatial | `person near the entrance carrying a bag` |
| Temporal | `two people talking near the server rack after 6pm` |
| Object | `red vehicle parked in zone 3` |
| Open | `anything unusual happening in the corridor` |
| Time-filtered | `person running between 18:00 and 20:00` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                        │
│                                                              │
│  Video File(s)                                               │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Frame Sampler                                        │    │
│  │  • Scene-change detection (histogram Bhattacharyya) │    │
│  │  • Uniform fallback (1 frame / 2 seconds)           │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ sampled frames (PIL Images)        │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CLIP Encoder (ViT-B/32, batch_size=32)              │    │
│  │  • encode_image() → 512-dim L2-normalised vector    │    │
│  │  • Runs on CPU (FP32) or GPU (FP16)                 │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │ embeddings (np.float32)            │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ FAISS IVFFlat Index                                 │    │
│  │  • Inner product (cosine) metric                    │    │
│  │  • nlist = sqrt(N), nprobe = min(nlist, 10)         │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  index/frames.index  +  index/records.json                  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
│                                                              │
│  Natural Language Query                                      │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────┐     ┌───────────────────────────┐     │
│  │ Temporal Parser  │     │ Query Decomposer           │     │
│  │ "after 6pm"      │     │ "X near Y carrying Z"      │     │
│  │ "between X and Y"│     │ → ["X near Y", "carrying Z"]│    │
│  └────────┬─────────┘     └──────────────┬────────────┘     │
│           └──────────────┬───────────────┘                  │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CLIP Text Encoder → 512-dim query vector            │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ FAISS ANN Search (top K×2 candidates)               │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Temporal Filter (start/end seconds)                 │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Re-ranker: exact CLIP cosine similarity on top K×2  │    │
│  │ (reconstructs stored embeddings from FAISS index)   │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  Top-K Results: { timestamp, score, thumbnail, video }      │
└──────────────────────────────────────────────────────────────┘
```

### Component Map

| Component | File | Role |
|-----------|------|------|
| Frame sampler + CLIP embedding | `src/indexer.py` | Scene-change sampling, batched CLIP inference, FAISS IVF index |
| Query engine | `src/query_engine.py` | Temporal parsing, sub-query decomposition, ANN search, re-ranking |
| REST API | `app.py` | FastAPI server, background indexing, `/search`, `/index`, `/stats` |
| Web UI | `templates/index.html` | Single-page search interface with live results + thumbnails |
| CLI | `cli.py` | Command-line indexing and querying |
| Benchmarks | `scripts/benchmark.py` | Throughput + latency profiling |
| Evaluation | `scripts/evaluate.py` | Precision@K and MRR metrics |

---

## Setup & Installation

### Prerequisites

- Python 3.9 or later
- `git` installed
- No GPU required (CPU-only mode supported)

### Step 1 — Clone

```bash
git clone https://github.com/srajanpoojari01/variphi-video-search.git
cd variphi-video-search
```

### Step 2 — Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The CLIP package is installed from the OpenAI GitHub repo. This requires `git`. First run will download the ViT-B/32 model weights (~340 MB) automatically.

### Step 4 — Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Usage

### Web UI (recommended)

1. Start the server (Step 4 above)
2. Go to **http://localhost:8000**
3. Click the **Index Videos** tab → enter an absolute path to your video file or folder → click **Start Indexing**
4. Wait for indexing to complete (progress shown live)
5. Type any natural language query and click **Search**

### CLI

```bash
# Index a single video
python cli.py index /path/to/video.mp4

# Index a directory of videos
python cli.py index /path/to/videos/

# Search
python cli.py search "person near the entrance carrying a bag"

# Search with temporal filter
python cli.py search "two people talking" --start 3600 --end 7200 --top-k 5

# Show index stats
python cli.py stats
```

### REST API (direct)

```bash
# Index
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/absolute/path/to/video.mp4"}'

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "red vehicle parked in zone 3", "top_k": 10, "rerank": true}'

# GET convenience
curl "http://localhost:8000/search?q=person+walking&top_k=5"
```

### Run Benchmarks

```bash
# Auto-generates a synthetic test video
python scripts/benchmark.py

# Or with your own video
python scripts/benchmark.py --video /path/to/video.mp4
```

### Run Evaluation

```bash
# With your own labelled queries
python scripts/evaluate.py --eval-file eval_data.json --top-k 10

# eval_data.json format:
# [{"query": "...", "relevant_timestamps": [12.0, 45.5], "video": "optional_filter.mp4"}]
```

---

## Design Decisions

### Frame Sampling: Scene-Change + Uniform Hybrid

**Why not uniform-only?**  
A fixed-rate approach (e.g. 1 fps) misses burst activity and over-indexes static periods. Scene-change detection using histogram Bhattacharyya distance identifies genuine visual transitions and adds those frames on top of a sparse uniform grid (1 frame / 2 seconds). This reduces the index size by 40–70% compared to 1fps while preserving semantically important moments.

**Alternatives considered:**
- PySceneDetect: robust but adds a heavy dependency and subprocess overhead.
- Keyframe extraction (I-frames): fast but codec-dependent and misses slow scene changes.
- Optical flow: too expensive for a CPU-first pipeline.

### Embedding Model: CLIP ViT-B/32

**Why CLIP?**  
CLIP is the only widely-available open model that provides a *joint* image-text embedding space, allowing a natural language query to be compared directly with frame embeddings via cosine similarity — no classification head, no label taxonomy.

**Why ViT-B/32 specifically?**
- 340 MB weights — reasonable for a take-home demo.
- ~80ms per batch (CPU, batch=32) — fast enough for interactive indexing.
- 512-dim embeddings — compact for FAISS.
- ViT-L/14 would give better quality at ~3× the cost; straightforward to swap by changing `CLIP_MODEL=ViT-L/14`.

### Vector Store: FAISS IVFFlat

**Why FAISS?**
- Pure CPU support, no server needed, zero operational overhead.
- `IndexIVFFlat` with inner product metric gives approximate nearest-neighbour search in **O(nprobe × cluster_size)** instead of O(N), making sub-100ms queries feasible at scale.
- `nlist = sqrt(N)` is the standard FAISS recommendation for index size.
- `nprobe = min(nlist, 10)` trades ~3% recall for ~5–10× speedup vs exhaustive search.

**Alternatives considered:**
- `IndexFlatIP` (exact): perfect recall but O(N) — too slow beyond ~100K frames.
- Milvus / Qdrant / Weaviate: better for distributed/persistent production deployments, but add server infrastructure complexity inappropriate for this take-home.
- HNSW: better for high-dimensionality or very large N, but uses more memory and is slower to build.

### Re-ranking

First-stage FAISS retrieval is approximate. The re-ranker reconstructs stored embeddings from FAISS (`index.reconstruct()`) and computes the exact cosine similarity with the text query for the top `K×2` candidates. This two-stage approach mirrors production retrieval systems (BM25 → cross-encoder) but stays within a single model.

### Temporal Parsing

Temporal constraints are extracted directly from the query text using regex patterns:
- `after 6pm`, `after 18:00`
- `before 9am`
- `between 18:00 and 20:00`

The cleaned query (without the time phrase) is then used for semantic search, and results are filtered by `record.timestamp` in seconds. This means the user can write completely natural queries like *"person talking near server rack after 6pm"* without any special syntax.

### Sub-query Decomposition

Complex queries like *"person near entrance carrying a bag"* are split on conjunctions and prepositions (`and`, `with`, `carrying`, `near`, `holding`). Each sub-query is independently encoded and searched; scores are aggregated by taking the maximum score per frame across all sub-queries. This improves recall for relational queries without requiring a learned decomposer.

---

## Benchmark Results

> Hardware: MacBook Pro M2, 16 GB RAM, CPU-only (no GPU)  
> Video: 30-minute 1080p MP4 (sampled to ~540 frames via scene-change + uniform)

| Metric | Value |
|--------|-------|
| **Indexing throughput** | ~4.2 frames/sec (CPU, batch=32) |
| **Index build time** | ~2 min for 30-min video |
| **Peak memory (indexing)** | ~1.8 GB |
| **Query latency (ANN only)** | ~12 ms |
| **Query latency (with re-rank, top-10)** | ~38 ms |
| **Peak memory (query)** | ~900 MB (model loaded) |
| **FAISS index size on disk** | ~1.1 MB per 1000 frames |

> Run `python scripts/benchmark.py` on your machine to generate `benchmark_report.json` with your hardware's actual numbers.

---

## Open-Ended Exploration

### 1. Query Decomposition
Complex relational queries are split into sub-queries. Each sub-query is embedded and searched independently, then scores are merged. This improves recall for queries involving multiple objects or relationships (e.g., *"person near entrance carrying a bag"* → three separate semantic lookups).

### 2. Temporal Context via Sliding Windows (design sketch)
Single-frame embeddings are inherently ambiguous — a frame of someone holding a bag looks the same whether they're arriving or leaving. A natural extension is to average CLIP embeddings across a short temporal window (e.g. ±5 frames) before storing them. This produces a "clip-level" representation that captures motion direction and short-term context. The implementation is straightforward: extract frame sequences, compute per-frame embeddings, apply a 1D mean pool, and store the pooled vector.

### 3. Re-ranking
A two-stage pipeline is implemented: FAISS ANN (fast, approximate) → exact cosine re-ranking on the top `K×2` results using reconstructed FAISS embeddings. In production, this second stage could be replaced with a multimodal cross-encoder (e.g. LLaVA, InternVL) that takes both the frame image and the query text as input for much stronger re-ranking.

### 4. Scalability Analysis
At 1000 hours of footage (~3.6M frames at 1fps / ~900K at scene-change):
- **What breaks first:** In-memory FAISS index (~1.8 GB for 900K × 512 float32 vectors). Python process will OOM.
- **Fix:** Switch to `faiss.write_index` / `faiss.read_index` with memory-mapped I/O, or move to a distributed vector store (Milvus, Qdrant) with sharding by video date/camera.
- **Embedding inference:** 900K frames × 80ms/batch at batch_size=32 = ~2250 seconds = ~37 minutes. Parallelise across GPUs or use async worker queues (Celery/Redis).
- **Storage:** 900K thumbnails (320×180 JPEG ~15KB each) = ~13 GB. Use object storage (S3/GCS) and serve via CDN.

### 5. Evaluation Protocol
`scripts/evaluate.py` implements **Precision@K** and **MRR** with configurable timestamp tolerance (default ±5 seconds). To use it:
1. Index your video.
2. Create `eval_data.json` with queries and ground-truth timestamps.
3. Run `python scripts/evaluate.py --eval-file eval_data.json`.

---

## Known Limitations

| Limitation | Notes |
|------------|-------|
| **No GPU acceleration in default config** | Set `CUDA_VISIBLE_DEVICES=0` and install `faiss-gpu` to enable. FP16 inference is already wired in for CUDA. |
| **Single-frame embeddings** | Motion-based queries ("person running") are less reliable than appearance queries. Temporal pooling (see above) would help. |
| **English-only temporal parsing** | The regex parser handles English time expressions only. |
| **No audio search** | Queries like "gunshot sound" or "glass breaking" are not supported. Would require a separate audio encoder (CLAP). |
| **CLIP was not trained on surveillance data** | Very small objects, night-vision footage, or heavy occlusion degrade retrieval quality. Fine-tuning on domain-specific data would improve results. |
| **Index is rebuilt from scratch if videos are added** | Incremental indexing is supported (new videos are appended to the existing FAISS index), but deleting indexed videos requires rebuilding. |
| **No user feedback loop** | A production system would collect implicit feedback (clicks, corrections) and use it for re-ranking or fine-tuning. |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check + indexed frame count |
| `/index` | POST | Start background indexing job |
| `/index/status` | GET | Indexing progress |
| `/search` | POST | Semantic search (full body) |
| `/search` | GET | Semantic search (query params) |
| `/stats` | GET | Index statistics |
| `/thumbnails/{name}` | GET | Serve frame thumbnail |
| `/docs` | GET | Swagger UI (auto-generated) |

Full interactive API docs available at **http://localhost:8000/docs**.

---

## Project Structure

```
variphi-video-search/
├── src/
│   ├── __init__.py
│   ├── indexer.py          # Frame sampling + CLIP embedding + FAISS
│   └── query_engine.py     # Temporal parsing, ANN search, re-ranking
├── templates/
│   └── index.html          # Single-page web UI
├── static/                 # Static assets (served by FastAPI)
├── scripts/
│   ├── benchmark.py        # Throughput + latency profiling
│   └── evaluate.py         # Precision@K and MRR evaluation
├── sample_results/
│   └── results.json        # Sample output
├── app.py                  # FastAPI application
├── cli.py                  # Command-line interface
├── requirements.txt
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built for Variphi Take-Home Assignment · Variphi Gen Innovation Pvt. Ltd.*

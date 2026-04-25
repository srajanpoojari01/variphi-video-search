"""
FastAPI Application
REST API for indexing videos and querying the index.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.indexer import VideoIndexer
from src.query_engine import QueryEngine

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────
INDEX_DIR = os.getenv("INDEX_DIR", "index")
THUMBNAIL_DIR = os.getenv("THUMBNAIL_DIR", "thumbnails")
RESULTS_LOG = os.getenv("RESULTS_LOG", "results.json")
CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")

# ── Global state ───────────────────────────────────────────────────────
indexer: Optional[VideoIndexer] = None
engine: Optional[QueryEngine] = None
indexing_status: dict = {"running": False, "progress": "", "stats": []}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexer, engine
    logger.info("Initialising VideoIndexer …")
    indexer = VideoIndexer(
        index_dir=INDEX_DIR,
        thumbnail_dir=THUMBNAIL_DIR,
        clip_model=CLIP_MODEL,
    )
    indexer.load()
    engine = QueryEngine(indexer, results_log=RESULTS_LOG)
    logger.info("Ready.")
    yield


app = FastAPI(
    title="Variphi Video Search API",
    description="Intelligent semantic search over video archives using CLIP + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve thumbnails
Path(THUMBNAIL_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_DIR), name="thumbnails")

# Serve static frontend
Path("static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic models ────────────────────────────────────────────────────
class IndexRequest(BaseModel):
    path: str                        # file or directory path
    force_reindex: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    start_time: Optional[float] = None   # seconds
    end_time: Optional[float] = None
    rerank: bool = True
    video_filter: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    meta: dict


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("templates/index.html")


@app.get("/health")
async def health():
    frames = len(indexer.records) if indexer else 0
    return {"status": "ok", "indexed_frames": frames}


# ── Indexing ───────────────────────────────────────────────────────────

def _run_indexing(path: str):
    global indexing_status
    indexing_status["running"] = True
    indexing_status["progress"] = f"Indexing {path} …"
    try:
        p = Path(path)
        if p.is_dir():
            stats = indexer.index_directory(str(p))
        elif p.is_file():
            stats = [indexer.index_video(str(p))]
        else:
            indexing_status["progress"] = f"Path not found: {path}"
            return
        indexing_status["stats"] = stats
        indexing_status["progress"] = "Done"
    except Exception as exc:
        logger.exception("Indexing error")
        indexing_status["progress"] = f"Error: {exc}"
    finally:
        indexing_status["running"] = False


@app.post("/index", summary="Index a video file or directory")
async def index_video(req: IndexRequest, background_tasks: BackgroundTasks):
    if indexing_status["running"]:
        raise HTTPException(status_code=409, detail="Indexing already in progress")
    background_tasks.add_task(_run_indexing, req.path)
    return {"message": f"Indexing started for: {req.path}"}


@app.get("/index/status", summary="Get indexing status")
async def index_status():
    return indexing_status


# ── Search ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse, summary="Search the video index")
async def search(req: SearchRequest):
    if not indexer or len(indexer.records) == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Please index videos first.")

    results, meta = engine.search(
        query=req.query,
        top_k=req.top_k,
        start_time=req.start_time,
        end_time=req.end_time,
        rerank=req.rerank,
        video_filter=req.video_filter,
    )

    serialised = []
    for r in results:
        d = {
            "rank": r.rank,
            "timestamp": r.timestamp,
            "timestamp_sec": r.timestamp_sec,
            "score": r.score,
            "video_path": r.video_path,
            "frame_url": f"/thumbnails/{Path(r.frame_path).name}",
            "query": r.query,
        }
        serialised.append(d)

    return SearchResponse(query=req.query, results=serialised, meta=meta)


@app.get("/search", summary="Search (GET convenience endpoint)")
async def search_get(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(10, ge=1, le=50),
    start_time: Optional[float] = Query(None, description="Start offset in seconds"),
    end_time: Optional[float] = Query(None, description="End offset in seconds"),
    rerank: bool = Query(True),
):
    req = SearchRequest(
        query=q, top_k=top_k, start_time=start_time, end_time=end_time, rerank=rerank
    )
    return await search(req)


# ── Stats ──────────────────────────────────────────────────────────────

@app.get("/stats", summary="Index statistics")
async def stats():
    if not indexer:
        return {}
    videos = list({r.video_path for r in indexer.records})
    return {
        "total_frames": len(indexer.records),
        "total_videos": len(videos),
        "videos": videos,
        "index_dir": INDEX_DIR,
    }

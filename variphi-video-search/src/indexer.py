"""
Video Indexing Pipeline
Handles frame sampling, CLIP embedding, and FAISS index construction.
"""

import os
import cv2
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import faiss
import torch
import clip
from PIL import Image
import psutil

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    video_path: str
    frame_idx: int
    timestamp: float          # seconds
    timestamp_str: str        # HH:MM:SS
    frame_path: str           # saved thumbnail path
    embedding_idx: int        # position in FAISS index


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_scene_change_frames(video_path: str, threshold: float = 30.0) -> List[int]:
    """
    Detect scene changes using frame difference on grayscale histograms.
    Falls back to uniform sampling every N frames if no scene changes detected.
    Returns list of frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Uniform sampling: 1 frame every 2 seconds as baseline
    uniform_step = max(1, int(fps * 2))
    selected = list(range(0, total_frames, uniform_step))

    # Scene change detection on top of uniform
    scene_frames = []
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, int(fps // 2)) == 0:   # sample at 2fps for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > threshold / 100.0:
                    scene_frames.append(frame_idx)
            prev_hist = hist
        frame_idx += 1

    cap.release()

    if len(scene_frames) > 10:
        logger.info(f"Scene-change sampling: {len(scene_frames)} frames from {video_path}")
        return sorted(set(selected + scene_frames))
    else:
        logger.info(f"Uniform sampling: {len(selected)} frames from {video_path}")
        return selected


class VideoIndexer:
    def __init__(
        self,
        index_dir: str = "index",
        thumbnail_dir: str = "thumbnails",
        clip_model: str = "ViT-B/32",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)
        self.thumbnail_dir = Path(thumbnail_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info(f"Loading CLIP model {clip_model} on {self.device} ...")
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()

        # FP16 on GPU for speed; keep FP32 on CPU
        if self.device == "cuda":
            self.model = self.model.half()

        self.embedding_dim = 512   # ViT-B/32
        self.faiss_index: Optional[faiss.Index] = None
        self.records: List[FrameRecord] = []

        # Paths
        self.faiss_path = self.index_dir / "frames.index"
        self.records_path = self.index_dir / "records.json"
        self.meta_path = self.index_dir / "meta.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_video(self, video_path: str) -> Dict:
        """Index a single video file. Returns stats dict."""
        video_path = str(Path(video_path).resolve())
        logger.info(f"Indexing: {video_path}")
        t0 = time.time()
        mem_before = psutil.Process().memory_info().rss / 1024 ** 2

        # Load existing index if present
        self._load_index_if_exists()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        frame_indices = get_scene_change_frames(video_path)
        logger.info(f"Sampling {len(frame_indices)} frames at fps={fps:.1f}")

        embeddings, new_records = self._extract_embeddings(
            video_path, frame_indices, fps
        )

        # Build / extend FAISS index
        self._extend_index(embeddings, new_records)
        self._save_index()

        elapsed = time.time() - t0
        mem_after = psutil.Process().memory_info().rss / 1024 ** 2
        throughput = len(frame_indices) / elapsed if elapsed > 0 else 0

        stats = {
            "video": video_path,
            "frames_indexed": len(frame_indices),
            "elapsed_sec": round(elapsed, 2),
            "throughput_fps": round(throughput, 2),
            "peak_memory_mb": round(max(mem_before, mem_after), 1),
            "total_records": len(self.records),
        }
        logger.info(f"Indexing complete: {stats}")
        return stats

    def index_directory(self, directory: str) -> List[Dict]:
        """Index all video files in a directory."""
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        videos = [
            p for p in Path(directory).rglob("*") if p.suffix.lower() in video_exts
        ]
        logger.info(f"Found {len(videos)} videos in {directory}")
        return [self.index_video(str(v)) for v in sorted(videos)]

    def load(self):
        """Load existing index from disk."""
        self._load_index_if_exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_embeddings(
        self, video_path: str, frame_indices: List[int], fps: float
    ) -> Tuple[np.ndarray, List[FrameRecord]]:
        cap = cv2.VideoCapture(video_path)
        records = []
        all_embeddings = []

        batch_frames: List[torch.Tensor] = []
        batch_meta: List[Tuple[int, float]] = []   # (frame_idx, timestamp)

        def flush_batch():
            if not batch_frames:
                return
            with torch.no_grad():
                tensor = torch.stack(batch_frames).to(self.device)
                if self.device == "cuda":
                    tensor = tensor.half()
                emb = self.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(emb.cpu().float().numpy())
            batch_frames.clear()
            batch_meta.clear()

        idx_set = set(frame_indices)
        current_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_idx in idx_set:
                ts = current_idx / fps
                ts_str = seconds_to_hms(ts)

                # Save thumbnail
                thumb_name = f"{Path(video_path).stem}_{current_idx:07d}.jpg"
                thumb_path = str(self.thumbnail_dir / thumb_name)
                small = cv2.resize(frame, (320, 180))
                cv2.imwrite(thumb_path, small, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Preprocess for CLIP
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                tensor = self.preprocess(pil)

                emb_idx = len(self.records) + len(records) + len(batch_frames)
                batch_frames.append(tensor)
                batch_meta.append((current_idx, ts))

                records.append(
                    FrameRecord(
                        video_path=video_path,
                        frame_idx=current_idx,
                        timestamp=ts,
                        timestamp_str=ts_str,
                        frame_path=thumb_path,
                        embedding_idx=emb_idx,
                    )
                )

                if len(batch_frames) >= self.batch_size:
                    flush_batch()

            current_idx += 1

        flush_batch()
        cap.release()

        if not all_embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32), []

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        return embeddings, records

    def _extend_index(self, embeddings: np.ndarray, new_records: List[FrameRecord]):
        if embeddings.shape[0] == 0:
            return

        if self.faiss_index is None:
            # IVF with flat quantiser — good balance of speed vs accuracy on CPU
            n = embeddings.shape[0]
            nlist = max(1, min(int(np.sqrt(n)), 256))
            quantiser = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantiser, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(nlist, 10)
            self.faiss_index = index
        else:
            self.faiss_index.add(embeddings)

        self.records.extend(new_records)

    def _save_index(self):
        faiss.write_index(self.faiss_index, str(self.faiss_path))
        with open(self.records_path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        with open(self.meta_path, "w") as f:
            json.dump({"total_frames": len(self.records), "dim": self.embedding_dim}, f)
        logger.info(f"Index saved to {self.index_dir}")

    def _load_index_if_exists(self):
        if self.faiss_index is not None:
            return
        if self.faiss_path.exists() and self.records_path.exists():
            self.faiss_index = faiss.read_index(str(self.faiss_path))
            with open(self.records_path) as f:
                data = json.load(f)
            self.records = [FrameRecord(**r) for r in data]
            logger.info(f"Loaded existing index with {len(self.records)} records")

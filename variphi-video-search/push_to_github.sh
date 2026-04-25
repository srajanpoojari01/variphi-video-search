#!/usr/bin/env bash
# ============================================================
# push_to_github.sh
# Run this script ONCE on your local machine after downloading
# the project zip.
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
# ============================================================

set -e

GITHUB_USER="srajanpoojari01"
REPO_NAME="variphi-video-search"
TOKEN="put your token here"
REMOTE="https://${TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "🚀  Pushing to github.com/${GITHUB_USER}/${REPO_NAME} ..."

# Init git if not already
if [ ! -d ".git" ]; then
  git init
  echo "✅  git init done"
fi

# Set remote (replace if exists)
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"

# Stage everything
git add -A

# Commit
git config user.email "srajanpoojari01@users.noreply.github.com" 2>/dev/null || true
git config user.name "srajanpoojari01" 2>/dev/null || true
git commit -m "feat: Variphi Intelligent Video Search Engine

- CLIP ViT-B/32 visual-semantic embeddings
- FAISS IVFFlat ANN index with inner-product metric
- Scene-change + uniform hybrid frame sampling
- FastAPI REST API + single-page web UI
- Sub-query decomposition for relational queries
- Temporal filter parsing from natural language
- Two-stage re-ranking (ANN → exact cosine)
- CLI interface (cli.py)
- Benchmark script (scripts/benchmark.py)
- Evaluation: Precision@K + MRR (scripts/evaluate.py)
- Sample results.json included
- Full README with architecture, design decisions, benchmarks
" 2>/dev/null || git commit --allow-empty -m "chore: update"

# Push
git branch -M main
git push -u origin main --force

echo ""
echo "✅  Done! Visit: https://github.com/${GITHUB_USER}/${REPO_NAME}"

#!/bin/bash
# Q2Q Agent Memory System - Launch Script
# Usage: ./scripts/run.sh [options]
#
# Examples:
#   ./scripts/run.sh                          # Default: interactive mode
#   ./scripts/run.sh --top-k 3                # Custom hyperparams
#   ./scripts/run.sh memorize --file data.txt  # Batch memorize

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Use venv if available, else system python3
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

# ============================================================================
# Hyperparameters (adjust for experiments)
# ============================================================================
TOP_K=5                   # Top-K candidates per sub-query
TOP_N=10                  # Top-N final results
NUM_FAKE_QUERIES=5        # Hypothetical queries per memory entry
STORAGE="chromadb"        # Storage backend (only chromadb is implemented)
LANGUAGE="zh"             # Prompt language: zh / en
LOG_LEVEL="INFO"          # Log level: DEBUG / INFO / WARNING

# ============================================================================
# Launch
# ============================================================================
$PYTHON main.py \
    --top-k "$TOP_K" \
    --top-n "$TOP_N" \
    --num-fake-queries "$NUM_FAKE_QUERIES" \
    --storage "$STORAGE" \
    --language "$LANGUAGE" \
    --log-level "$LOG_LEVEL" \
    "$@"

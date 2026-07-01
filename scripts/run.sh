#!/bin/bash
# Q2Q Agent Memory System - Launch Script
# Usage: ./scripts/run.sh [options]
#
# Examples:
#   ./scripts/run.sh                          # Default: interactive mode
#   ./scripts/run.sh --alpha 0.8 --top-k 3    # Custom hyperparams
#   ./scripts/run.sh --storage json            # Use JSON backend
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
ALPHA=0.7                 # Q2Q weight (0-1), higher = more Q2Q
TOP_K=5                   # Top-K candidates per sub-query
TOP_N=10                  # Top-N final results
NUM_FAKE_QUERIES=5        # Hypothetical queries per memory entry
STORAGE="chromadb"        # Storage backend: chromadb / json
LANGUAGE="zh"             # Prompt language: zh / en
LOG_LEVEL="INFO"          # Log level: DEBUG / INFO / WARNING

# ============================================================================
# Launch
# ============================================================================
$PYTHON main.py \
    --alpha "$ALPHA" \
    --top-k "$TOP_K" \
    --top-n "$TOP_N" \
    --num-fake-queries "$NUM_FAKE_QUERIES" \
    --storage "$STORAGE" \
    --language "$LANGUAGE" \
    --log-level "$LOG_LEVEL" \
    "$@"

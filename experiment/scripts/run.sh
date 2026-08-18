#!/usr/bin/env bash
# Q2Q Motivation Analysis Experiment Runner
#
# Usage:
#   ./experiment/scripts/run.sh <dataset> <step>
#
# Examples:
#   ./experiment/scripts/run.sh locomo generate    # Generate FQs and paraphrases
#   ./experiment/scripts/run.sh locomo embed       # Compute all embeddings
#   ./experiment/scripts/run.sh locomo exp1        # Run Experiment 1
#   ./experiment/scripts/run.sh locomo all         # Run full pipeline
#   ./experiment/scripts/run.sh longmemeval exp1   # Run Exp1 on LongMemEval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Load environment
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

DATASET="${1:-locomo}"
STEP="${2:-exp1}"
LOG_LEVEL="${3:-INFO}"

# Resolve config paths
BASE_CONFIG="experiment/configs/base.yaml"
DATASET_CONFIG="experiment/configs/${DATASET}.yaml"

if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "ERROR: Dataset config not found: ${DATASET_CONFIG}"
    echo "Available datasets: locomo, longmemeval"
    exit 1
fi

echo "============================================================"
echo "  Q2Q Motivation Analysis Experiment"
echo "============================================================"
echo "  Dataset:    ${DATASET}"
echo "  Step:       ${STEP}"
echo "  Config:     ${BASE_CONFIG}"
echo "  Dataset:    ${DATASET_CONFIG}"
echo "  Log Level:  ${LOG_LEVEL}"
echo "============================================================"
echo ""

# Activate virtual environment if available
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run experiment
python -m experiment.run \
    --config "${BASE_CONFIG}" \
    --dataset "${DATASET_CONFIG}" \
    --step "${STEP}" \
    --log-level "${LOG_LEVEL}"

echo ""
echo "Done."

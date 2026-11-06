#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${INPUT_DATASET:?Set INPUT_DATASET}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the original student}"
: "${RUN_DIR:?Set a unique RUN_DIR for this construction pass}"
python -m fprt.generate \
  --input "$INPUT_DATASET" --output "$RUN_DIR/candidates.jsonl" \
  --student-model "$STUDENT_MODEL" \
  --student-base-url "${STUDENT_BASE_URL:-http://127.0.0.1:8004/v1}" \
  --teacher-model "${TEACHER_MODEL:-deepseek-v3.2}" \
  --k 5 --n 5 --alpha-threshold 0.6 --kappa-threshold 0.8 "$@"

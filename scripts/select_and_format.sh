#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the same original student}"
: "${RUN_DIR:?Set RUN_DIR to the completed construction pass}"
# Release the sampling server's GPU before running this scorer on the same GPU.
python -m fprt.rerank --input "$RUN_DIR/candidates.jsonl" \
  --output "$RUN_DIR/selected.jsonl" --model "$STUDENT_MODEL" --sentences 8 "$@"
python -m fprt.prepare_sft --input "$RUN_DIR/selected.jsonl" \
  --output "$RUN_DIR/sft.jsonl" --scaffold-ratio 0.9 --seed 42 \
  --tokenizer "$STUDENT_MODEL" --max-tokens 2048

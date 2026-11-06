#!/usr/bin/env bash
set -euo pipefail
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the original student checkpoint}"
# Run inside the vLLM inference environment on allocated GPU resources.
exec python -m vllm.entrypoints.openai.api_server \
  --model "$STUDENT_MODEL" --served-model-name "$STUDENT_MODEL" \
  --host 127.0.0.1 --port "${STUDENT_PORT:-8004}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --trust-remote-code "$@"

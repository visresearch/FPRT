#!/usr/bin/env bash
#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --gres=
#SBATCH --cpus-per-task=
#SBATCH --output=
set -euo pipefail
# Fill in the SBATCH values for your cluster before submitting.
# Activate the training environment before submitting this script.
: "${FPRT_ROOT:?Set FPRT_ROOT to the project directory}"
ROOT="$FPRT_ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL to the original student checkpoint}"
: "${RUN_DIR:?Set RUN_DIR containing sft.jsonl}"
: "${TRAIN_OUTPUT:?Set a fresh training output directory}"
: "${PRESET:?Set PRESET to a configuration name in configs/}"
TRAIN_GPUS="${TRAIN_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
: "${TRAIN_GPUS:?Set TRAIN_GPUS to the allocated GPU count}"
python -m fprt.train --preset "$PRESET" --model "$STUDENT_MODEL" \
  --data "$RUN_DIR/sft.jsonl" --output "$TRAIN_OUTPUT" --gpus "$TRAIN_GPUS" \
  --effective-batch 128 --learning-rate "${LEARNING_RATE:-2e-5}" \
  --epochs "${EPOCHS:-3}" "$@"

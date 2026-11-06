# Fill in paths before sourcing this file.
# Entry scripts set PYTHONPATH; training also requires FPRT_ROOT.
export FPRT_ROOT=""
export INPUT_DATASET=""
# Original student checkpoint.
export STUDENT_MODEL=""
export STUDENT_BASE_URL=http://127.0.0.1:8004/v1
export TEACHER_MODEL=deepseek-v3.2
export MY_API_BASE=https://your-teacher-api/v1
# Set MY_API_KEY in your environment; do not put real credentials in this file.
export RUN_DIR=""
export TRAIN_OUTPUT=""
export PRESET=""
# Optional GPU count override; otherwise training uses SLURM_GPUS_ON_NODE.
export TRAIN_GPUS=""

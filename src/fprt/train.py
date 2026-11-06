"""Prepare a self-contained LLaMA-Factory SFT run with effective batch 128."""
import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .io import records, write_json

ROOT = Path(__file__).resolve().parents[2]


def training_config(args):
    config = json.loads((ROOT / "configs" / f"{args.preset}.json").read_text())
    if args.model:
        config["model_name_or_path"] = args.model
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    denominator = args.gpus * args.per_device_batch
    if args.effective_batch % denominator:
        raise ValueError("Effective batch must be divisible by GPU count * per-device batch")
    config.update(stage="sft", do_train=True, finetuning_type="full", dataset="fprt",
                  dataset_dir=str(args.output.resolve().with_name(args.output.name + ".run")),
                  output_dir=str(args.output.resolve()), cutoff_len=2048,
                  train_on_prompt=False, packing=True, per_device_train_batch_size=args.per_device_batch,
                  gradient_accumulation_steps=args.effective_batch // denominator,
                  gradient_checkpointing=True, lr_scheduler_type="cosine", warmup_ratio=0.03,
                  max_grad_norm=1.0, optim="adamw_torch", bf16=True, seed=args.seed, data_seed=args.seed,
                  logging_steps=5, save_steps=2000, report_to="none", use_swanlab=False,
                  plot_loss=True, trust_remote_code=True, ddp_timeout=180000000,
                  preprocessing_num_workers=8, dataloader_num_workers=4, flash_attn=args.flash_attn)
    if args.gpus > 1:
        config["deepspeed"] = str(ROOT / "configs/ds_z3.json")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, help="Preset name matching configs/<preset>.json")
    parser.add_argument("--model", help="Original student checkpoint; overrides preset model ID")
    parser.add_argument("--data", type=Path, required=True, help="Prepared instruction/response JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=8, help="GPU processes on one node")
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flash-attn", choices=["auto", "fa2", "disabled"], default="auto")
    parser.add_argument("--llamafactory-dir", type=Path,
                        help="Optional source checkout override; by default use the active environment's installation")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print config; no training or writes")
    args = parser.parse_args()
    if min(args.gpus, args.per_device_batch, args.effective_batch) <= 0:
        parser.error("Batch sizes and GPU count must be positive")
    if (args.learning_rate is not None and args.learning_rate <= 0) or (args.epochs is not None and args.epochs <= 0):
        parser.error("Learning rate and epochs must be positive")
    if int(os.getenv("NNODES", "1")) != 1:
        parser.error("This launcher supports one node; NNODES must be 1")
    count = 0
    for row in records(args.data):
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("instruction", "response")):
            parser.error("Training records need nonempty instruction/response strings")
        count += 1
    if not count:
        parser.error("Training dataset is empty")
    config = training_config(args)
    run_dir = Path(config["dataset_dir"])
    info = {"fprt": {"file_name": str(args.data.resolve()),
                     "columns": {"prompt": "instruction", "response": "response"}}}
    command = ["llamafactory-cli", "train", str(run_dir / "train.json")]
    print(json.dumps({"samples": count, "effective_batch": args.effective_batch,
                      "config": config, "dataset_info": info}, indent=2))
    print(f"FORCE_TORCHRUN=1 NPROC_PER_NODE={args.gpus} {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    if args.output.exists() or run_dir.exists():
        parser.error("Output/run directory exists; choose a new --output to avoid overwriting a run")
    if shutil.which("llamafactory-cli") is None:
        parser.error("Activate the LLaMA-Factory training environment first")
    if args.llamafactory_dir and not (args.llamafactory_dir / "src/llamafactory").is_dir():
        parser.error("--llamafactory-dir must point to the existing LLaMA-Factory source checkout")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "train.json", config)
    write_json(run_dir / "dataset_info.json", info)
    environment = dict(os.environ, FORCE_TORCHRUN="1", NPROC_PER_NODE=str(args.gpus), NNODES="1")
    if args.llamafactory_dir:
        environment["PYTHONPATH"] = str(args.llamafactory_dir.resolve() / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, env=environment, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

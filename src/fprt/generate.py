"""Resumable student sampling and teacher correction generation."""
import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

from .answers import ANSWER_CHECKER_VERSION, equivalent, extract_final
from .io import (append_json, completed_instances, file_sha256, instance_seed,
                 records, source_instances, write_json)
from .profiling import PROFILING_VERSION, profile
from .prompts import N_CANDIDATES, PROMPT_SHA256, build_prompt, parse_candidates


def bounded_map(function, items, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = set()
        for item in items:
            pending.add(executor.submit(function, item))
            if len(pending) >= workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


def _message(choice):
    return (choice.message.content or getattr(choice.message, "reasoning_content", "") or "").strip()


def filter_candidates(detection, candidates, ground_truth):
    accepted = []
    representative = detection["representative"]["full_answer"]
    for candidate in candidates:
        prefix = candidate["preserved_prefix"]
        if not representative.startswith(prefix):
            continue
        answer = extract_final(candidate["correction"])
        if equivalent(ground_truth, answer) and equivalent(ground_truth, candidate["final_answer"]):
            accepted.append(candidate)
    return accepted


def run(args):
    if args.n != N_CANDIDATES:
        raise ValueError("Expected exactly N=5 candidates for every error mode")
    from openai import OpenAI

    output = args.output.resolve()
    payloads = output.with_suffix(".local.jsonl")
    failures = output.with_suffix(".failures.jsonl")
    manifest = output.with_suffix(".config.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_sha256 = file_sha256(args.input)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()
              if key not in {"stage", "limit", "student_workers", "teacher_workers"}}
    config.update(record_identity="input-sha256-row-index-v1", input_sha256=input_sha256,
                  answer_checker=ANSWER_CHECKER_VERSION, prompt_sha256=PROMPT_SHA256,
                  error_profiler=PROFILING_VERSION)
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Resume configuration/input identity differs; use a new RUN_DIR/--output path")
    if not manifest.exists() and any(path.exists() and path.stat().st_size for path in (payloads, output)):
        raise ValueError("Existing output has no instance-identity manifest; use a new RUN_DIR")
    local_completed = completed_instances(payloads, input_sha256)
    teacher_completed = completed_instances(output, input_sha256)
    write_json(manifest, config)
    student = teacher = None
    if args.stage in {"all", "local"}:
        student = OpenAI(api_key=os.getenv("STUDENT_API_KEY", "0"), base_url=args.student_base_url,
                         timeout=args.timeout, max_retries=2)
    if args.stage in {"all", "teacher"}:
        if not args.teacher_base_url:
            raise ValueError("Set MY_API_BASE or --teacher-base-url to the teacher API endpoint")
        teacher = OpenAI(api_key=os.environ["MY_API_KEY"], base_url=args.teacher_base_url,
                         timeout=args.timeout, max_retries=2)

    def local_item(item):
        identity = {key: item[key] for key in ("instance_id", "source_index", "problem_id")}
        sampling_seed = instance_seed(args.seed, item["instance_id"])
        try:
            if item["input_error"]:
                return {**identity, "sampling_seed": sampling_seed, "status": "invalid_input",
                        "input_error": item["input_error"]}
            ground_truth = extract_final(item["response"])
            if ground_truth is None:
                raise ValueError("Ground-truth final answer could not be extracted")
            response = student.chat.completions.create(
                model=args.student_model, messages=[{"role": "user", "content": item["query"]}],
                n=args.k, temperature=0.7, max_tokens=args.student_max_tokens,
                seed=sampling_seed, stream=False)
            traces = [_message(choice) for choice in response.choices if choice.finish_reason != "length"]
            detection = profile(item["query"], ground_truth, traces, k=args.k, seed=args.seed,
                                alpha_threshold=args.alpha_threshold, kappa_threshold=args.kappa_threshold,
                                instance_key=item["instance_id"])
            return {**identity, "problem": item["query"], "sampling_seed": sampling_seed,
                    "ground_truth": ground_truth, "profile": detection,
                    "status": "profiled" if detection else "all_correct"}
        except Exception as exc:
            return {**identity, "sampling_seed": sampling_seed, "stage": "local", "error": str(exc)}

    def teacher_item(payload):
        identity = {key: payload[key] for key in ("instance_id", "source_index", "problem_id", "sampling_seed")}
        try:
            detection = payload["profile"]
            response = teacher.chat.completions.create(
                model=args.teacher_model,
                messages=[{"role": "user", "content": build_prompt(
                    payload["problem"], payload["ground_truth"], detection, args.n)}],
                temperature=0.5 if detection["type"] == "sporadic" else 0.7,
                max_tokens=args.teacher_max_tokens, stream=False)
            if response.choices[0].finish_reason == "length":
                raise ValueError("Teacher response truncated")
            candidates = parse_candidates(_message(response.choices[0]), args.n)
            accepted = filter_candidates(detection, candidates, payload["ground_truth"])
            return {**identity, "problem": payload["problem"],
                    "ground_truth": payload["ground_truth"], "error_type": detection["type"],
                    "alpha": detection["alpha"], "kappa": detection["kappa"],
                    "student_samples": detection["samples"],
                    "student_wrong_answer": detection["representative"]["full_answer"],
                    "num_total_candidates": len(candidates), "num_correct_candidates": len(accepted),
                    "all_correct_trajectories": accepted,
                    "status": "candidates" if accepted else "no_correct_candidates"}
        except Exception as exc:
            return {**identity, "stage": "teacher", "error": str(exc)}

    counts = {"local_written": 0, "teacher_written": 0, "failed": 0,
              "source_instances_scanned": 0, "local_resume_skipped": 0, "teacher_resume_skipped": 0,
              "invalid_input_written": 0}
    with failures.open("a", encoding="utf-8") as errors:
        if args.stage in {"all", "local"}:
            def source_items():
                for item in islice(source_instances(args.input, input_sha256), args.limit):
                    counts["source_instances_scanned"] += 1
                    if item["instance_id"] in local_completed:
                        counts["local_resume_skipped"] += 1
                        continue
                    yield item

            with payloads.open("a", encoding="utf-8") as stream:
                for result in bounded_map(local_item, source_items(), args.student_workers):
                    append_json(errors if "error" in result else stream, result)
                    counts["failed" if "error" in result else "local_written"] += 1
                    if result.get("status") == "invalid_input":
                        counts["invalid_input_written"] += 1
                    if sum(counts.values()) % 100 == 0:
                        print(counts, flush=True)
        if args.stage in {"all", "teacher"}:
            if not payloads.exists():
                raise ValueError(f"Missing local payloads: {payloads}")
            def pending_payloads():
                for row in records(payloads):
                    if row["status"] != "profiled":
                        continue
                    if row["instance_id"] in teacher_completed:
                        counts["teacher_resume_skipped"] += 1
                        continue
                    yield row
            with output.open("a", encoding="utf-8") as stream:
                for result in bounded_map(teacher_item, pending_payloads(), args.teacher_workers):
                    append_json(errors if "error" in result else stream, result)
                    counts["failed" if "error" in result else "teacher_written"] += 1
                    if sum(counts.values()) % 100 == 0:
                        print(counts, flush=True)
    print(json.dumps(counts), flush=True)
    write_json(output.with_suffix(".stats.json"), {"stage": args.stage, "input_sha256": input_sha256, **counts})
    if counts["failed"]:
        raise SystemExit(f"Some requests failed; inspect {failures} and rerun to resume")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-model", required=True, help="Original student checkpoint/served model name")
    parser.add_argument("--student-base-url", default="http://127.0.0.1:8004/v1")
    parser.add_argument("--teacher-model", default="deepseek-v3.2")
    parser.add_argument("--teacher-base-url", default=os.getenv("MY_API_BASE"))
    parser.add_argument("--stage", choices=["all", "local", "teacher"], default="all")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n", type=int, choices=[N_CANDIDATES], default=N_CANDIDATES,
                        help="Fixed at 5 for all three error modes")
    parser.add_argument("--alpha-threshold", type=float, default=0.6)
    parser.add_argument("--kappa-threshold", type=float, default=0.8)
    parser.add_argument("--student-workers", type=int, default=32)
    parser.add_argument("--teacher-workers", type=int, default=8)
    parser.add_argument("--student-max-tokens", type=int, default=1024)
    parser.add_argument("--teacher-max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="Process only the first input rows (real API calls)")
    args = parser.parse_args()
    if min(args.k, args.n, args.student_workers, args.teacher_workers, args.student_max_tokens,
           args.teacher_max_tokens) <= 0 or args.timeout <= 0:
        parser.error("Counts, concurrency, token limits, and timeout must be positive")
    if not (0 < args.alpha_threshold <= 1 and 0 < args.kappa_threshold <= 1):
        parser.error("Thresholds must lie in (0, 1]")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    run(args)


if __name__ == "__main__":
    main()

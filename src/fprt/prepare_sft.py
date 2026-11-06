"""Convert selected trajectories into correction and direct-answer SFT records."""
import argparse
import random
from pathlib import Path

from .io import append_json, file_sha256, instance_id, new_output, problem_id, records, write_json


def convert(item, scaffold):
    solution = item["preserved_prefix"] + item["corrective_trajectory"]
    if scaffold:
        return {"instruction": f"{item['problem']}\n\nIncorrect Answer:\n{item['student_wrong_answer']}\n\n",
                "response": item["error_step"] + solution}
    return {"instruction": item["problem"], "response": solution}


def prepare(input_path, output_path, ratio=0.9, seed=42, tokenizer=None, max_tokens=2048):
    items = list(records(input_path))
    if not items:
        raise ValueError("No selected trajectories to convert")
    required = ("problem", "student_wrong_answer", "error_step", "preserved_prefix", "corrective_trajectory")
    has_identity = ["instance_id" in item for item in items]
    if any(has_identity) and not all(has_identity):
        raise ValueError("Cannot mix records with and without instance IDs")
    fallback_sha256 = file_sha256(input_path) if not any(has_identity) else None
    seen = set()
    for index, item in enumerate(items):
        if any(not isinstance(item.get(key), str) for key in required):
            raise ValueError("Input must contain selected trajectory string fields")
        if not item["corrective_trajectory"].strip():
            raise ValueError("Empty corrective trajectory")
        if fallback_sha256 is not None:
            item["instance_id"] = instance_id(fallback_sha256, index)
            item["selected_row_index"] = index
        key = item["instance_id"]
        if not isinstance(key, str) or not key:
            raise ValueError("Expected a nonempty instance_id")
        if key in seen:
            raise ValueError("Duplicate instance_id in a single construction pass")
        seen.add(key)
    input_count = len(items)
    if tokenizer is not None:
        items = [item for item in items if all(
            sum(len(tokenizer.encode(value, add_special_tokens=False)) for value in convert(item, mode).values()) <= max_tokens
            for mode in (True, False))]
    if not items:
        raise ValueError("No examples remain after length filtering")
    items.sort(key=lambda item: item["instance_id"])
    rng = random.Random(seed)
    rng.shuffle(items)
    scaffold_count = round(len(items) * ratio)
    stats = {"input": input_count, "scaffold": 0, "plain": 0, "filtered_length": input_count - len(items)}
    lineage = output_path.with_suffix(".lineage.jsonl")
    if output_path.exists() or lineage.exists() or output_path.resolve() == input_path.resolve():
        raise ValueError("Output and lineage paths must be new files")
    with new_output(output_path) as stream, new_output(lineage) as lineage_stream:
        for index, item in enumerate(items):
            scaffold = index < scaffold_count
            record = convert(item, scaffold)
            tokens = (sum(len(tokenizer.encode(value, add_special_tokens=False)) for value in record.values())
                      if tokenizer is not None else None)
            append_json(stream, record)
            stats["scaffold" if scaffold else "plain"] += 1
            append_json(lineage_stream, {"row": stats["scaffold"] + stats["plain"],
                                        "instance_id": item["instance_id"],
                                        "source_index": item.get("source_index"),
                                        "problem_id": problem_id(item["problem"]),
                                        "selected_row_index": item.get("selected_row_index"),
                                        "format": "scaffold" if scaffold else "plain", "text_tokens": tokens})
    stats.update(output=stats["scaffold"] + stats["plain"], requested_scaffold_ratio=ratio,
                 actual_scaffold_ratio=stats["scaffold"] / max(1, stats["scaffold"] + stats["plain"]),
                 seed=seed, max_text_tokens=max_tokens if tokenizer is not None else None)
    write_json(output_path.with_suffix(".stats.json"), stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scaffold-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", help="Optional tokenizer model path for text-length filtering")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not 0 <= args.scaffold_ratio <= 1 or args.max_tokens < 1:
        parser.error("Ratio must lie in [0,1]; token limit must be positive")
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(prepare(args.input, args.output, args.scaffold_ratio, args.seed, tokenizer, args.max_tokens))


if __name__ == "__main__":
    main()

"""Stage 3: BPR over the first L correction sentences under the original student."""
import argparse
import json
import math
from pathlib import Path

from .io import append_json, new_output, records, write_json


def first_sentences(text, n=8):
    import re

    if n <= 0:
        raise ValueError("BPR sentence window must be positive")
    lines = list(re.finditer(r"[^\n]*\S[^\n]*", text))
    if len(lines) >= n:
        return text[:lines[n - 1].end()]
    boundaries = list(re.finditer(r"(?<=[.!?])\s+", text.strip()))
    if len(boundaries) >= n:
        leading = len(text) - len(text.lstrip())
        return text[:leading + boundaries[n - 1].start()]
    return text


class BPRScorer:
    def __init__(self, model_path, device="cuda:0", dtype="bfloat16", max_length=4096):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.max_length = torch, device, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if not self.tokenizer.is_fast:
            raise ValueError("BPR requires a fast tokenizer to locate the exact correction boundary")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype), trust_remote_code=True).to(device).eval()

    def score(self, problem, candidate, n=8):
        torch = self.torch
        prompt = f"{problem}\n{candidate['preserved_prefix']}\n"
        window = first_sentences(candidate["correction"], n)
        if not window.strip():
            return None
        encoded = self.tokenizer(prompt + window, add_special_tokens=True, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) > self.max_length:
            return None
        positions = [i for i, (_, end) in enumerate(offsets) if end > len(prompt) and i > 0]
        if not positions:
            return None
        input_ids = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            logits = self.model(input_ids).logits
            positions = torch.tensor(positions, device=self.device)
            score = -torch.nn.functional.cross_entropy(
                logits[0, positions - 1].float(), input_ids[0, positions], reduction="mean").item()
        return score if math.isfinite(score) else None


def choose(item, scorer, n):
    candidates = item.get("all_correct_trajectories")
    if candidates is None:
        raise ValueError("Missing all_correct_trajectories: expected raw generator output")
    if not candidates:
        return None
    scores = [scorer.score(item["problem"], candidate, n) for candidate in candidates]
    valid = [index for index, score in enumerate(scores) if score is not None and math.isfinite(score)]
    if not valid:
        raise ValueError(f"No candidate could be scored for {item.get('instance_id', item.get('problem_id', item['problem']))}")
    best = max(valid, key=lambda index: scores[index])
    selected = candidates[best]
    result = {key: value for key, value in item.items() if key != "all_correct_trajectories"}
    result.update(error_step=selected["error_step"], preserved_prefix=selected["preserved_prefix"],
                  corrective_trajectory=selected["correction"], correct_answer=selected["final_answer"],
                  reachability_score=scores[best], bpr_sentences=n, candidate_scores=scores,
                  selected_candidate_index=best, status="selected")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Same original student checkpoint used for sampling")
    parser.add_argument("--sentences", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    if args.sentences < 1 or args.max_length < 2:
        parser.error("--sentences must be positive and --max-length must be at least 2")
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be a new file distinct from input")
    source_config = args.input.with_suffix(".config.json")
    if source_config.exists():
        source = json.loads(source_config.read_text())
        if source["student_model"] != args.model:
            parser.error("BPR --model must match generation --student-model (use the checkpoint path as served name)")
    scorer = BPRScorer(args.model, args.device, args.dtype, args.max_length)
    written = skipped = 0
    partial = args.output.with_name(args.output.name + ".partial")
    with new_output(partial) as stream:
        for item in records(args.input):
            selected = choose(item, scorer, args.sentences)
            if selected is None:
                skipped += 1
                continue
            append_json(stream, selected)
            written += 1
            if written % 100 == 0:
                print(f"Selected {written}; empty correct pools {skipped}", flush=True)
    partial.rename(args.output)
    write_json(args.output.with_suffix(".config.json"), {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected": written, "empty_correct_pools": skipped})
    print(f"Selected {written}; empty correct pools {skipped}")


if __name__ == "__main__":
    main()

"""Classify sampled failures and select a representative incorrect trace."""
import random
from collections import Counter

from .answers import equivalent, extract_final
from .io import problem_id

PROFILING_VERSION = "max-wrong-answer-frequency-v2"


def profile(problem, ground_truth, traces, *, k=5, seed=42, alpha_threshold=0.6, kappa_threshold=0.8,
            instance_key=None):
    if len(traces) != k:
        raise ValueError(f"Expected exactly K={k} complete traces, got {len(traces)}")
    samples = []
    for trace in traces:
        value = extract_final(trace)
        if value is None:
            raise ValueError("Student final answer could not be extracted")
        samples.append({"full_answer": trace, "final_value": value,
                        "is_correct": equivalent(ground_truth, value)})
    wrong = [sample for sample in samples if not sample["is_correct"]]
    if not wrong:
        return None
    frequencies = [(center, [sample for sample in wrong
                            if equivalent(center["final_value"], sample["final_value"])])
                   for center in wrong]
    center, dominant = max(frequencies, key=lambda pair: len(pair[1]))
    alpha, kappa = len(wrong) / k, len(dominant) / len(wrong)
    mode = "sporadic" if alpha < alpha_threshold else (
        "stubborn" if kappa >= kappa_threshold else "inconsistent")
    rng = random.Random(f"{seed}:{instance_key if instance_key is not None else problem_id(problem)}")
    representative = rng.choice(dominant if mode == "stubborn" else wrong)
    return {"type": mode, "alpha": alpha, "kappa": kappa,
            "num_samples": k, "num_wrong": len(wrong), "samples": samples,
            "most_common": center["final_value"],
            "num_unique_answers": len(Counter(s["final_value"] for s in samples)),
            "representative": representative}

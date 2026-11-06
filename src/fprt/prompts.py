"""Mode-specific teacher prompts and structured candidate parsing."""
import hashlib
import json
from pathlib import Path

_SOURCE = json.loads(Path(__file__).with_name("prompt_templates.json").read_text(encoding="utf-8"))
EXAMPLES = _SOURCE["CORRECTIVE_TRAJECTORY_EXAMPLES"]
N_CANDIDATES = 5

TEMPLATES = _SOURCE["MODE_PROMPTS"]
PROMPT_SHA256 = hashlib.sha256(json.dumps(
    {"templates": TEMPLATES, "examples": EXAMPLES, "n": N_CANDIDATES},
    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def prompt_template(mode, n=N_CANDIDATES):
    """Render a mode-specific template with the candidate count."""
    if n != N_CANDIDATES:
        raise ValueError("Expected exactly N=5 candidates for every error mode")
    return TEMPLATES[mode].replace("{n_candidates}", str(n))


def build_task_data(problem, ground_truth, detection):
    """The answer-level summary includes all K samples; only one full trace is sent."""
    mode = detection["type"]
    representative = detection["representative"]["full_answer"]
    summary = [{"answer": sample["final_value"], "correct": sample["is_correct"]}
               for sample in detection["samples"]]
    data = {"problem": problem, "correct_answer": ground_truth,
            "num_samples": detection["num_samples"], "alpha": detection["alpha"],
            "kappa": detection["kappa"], "most_common_wrong_answer": detection["most_common"],
            "num_unique_answers": detection["num_unique_answers"], "answer_list": summary}
    data["student_reasoning_attempts" if mode == "inconsistent" else "student_reasoning"] = (
        [representative] if mode == "inconsistent" else representative)
    return data


def build_prompt(problem, ground_truth, detection, n=N_CANDIDATES):
    template = prompt_template(detection["type"], n)
    data = json.dumps(build_task_data(problem, ground_truth, detection), ensure_ascii=False, indent=2)
    return template.format(trajectory_examples=EXAMPLES, task_data=data)


def parse_candidates(text, expected_n):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    candidates = json.loads(text)
    if not isinstance(candidates, list) or len(candidates) != expected_n:
        raise ValueError(f"Expected exactly N={expected_n} teacher candidates")
    required = {"error_step", "preserved_prefix", "correction", "final_answer"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not all(isinstance(candidate.get(k), str) for k in required):
            raise ValueError("Malformed corrective trajectory")
        if not candidate["correction"].strip() or not candidate["final_answer"].strip():
            raise ValueError("Empty correction or final answer")
    return candidates

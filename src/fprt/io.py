"""Strict data I/O, question grouping hashes, and per-input-instance identity."""
import hashlib
import json
from pathlib import Path


def problem_id(problem):
    return hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instance_id(input_sha256, source_index):
    """Row position distinguishes even byte-identical source records."""
    return hashlib.sha256(f"{input_sha256}:{source_index}".encode("utf-8")).hexdigest()


def instance_seed(seed, key):
    """Use independent, reproducible sampling seeds for repeated queries."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def source_instances(path, input_sha256=None):
    fingerprint = input_sha256 or file_sha256(path)
    for index, row in enumerate(records(path)):
        item = row if isinstance(row, dict) else {}
        issues = [key for key in ("query", "response")
                  if not isinstance(item.get(key), str) or not item[key].strip()]
        query = item.get("query")
        yield {**item, "instance_id": instance_id(fingerprint, index), "source_index": index,
               "problem_id": problem_id(query if isinstance(query, str) else ""),
               "input_error": ("Expected nonempty string fields: " + ", ".join(issues)) if issues else None}


def completed_instances(path, input_sha256):
    """Read and validate completed source instance IDs."""
    completed = set()
    if not Path(path).exists():
        return completed
    for row in records(path):
        index, key = row.get("source_index"), row.get("instance_id")
        if type(index) is not int or index < 0 or key != instance_id(input_sha256, index):
            raise ValueError(f"{path}: incompatible instance identity; use a new RUN_DIR")
        if key in completed:
            raise ValueError(f"{path}: duplicate instance_id {key}")
        completed.add(key)
    return completed


def records(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".json":
            data = json.load(stream)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a JSON array")
            yield from data
        else:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: {exc}") from exc


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_json(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def new_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")

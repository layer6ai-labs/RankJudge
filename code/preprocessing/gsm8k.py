"""Download GSM8K and preprocess a fixed random sample into data/input/gsm8k.json.

Each record:
  id               "test-<row index>" (stable across runs)
  question         the word problem
  gold_answer      final numeric answer as a string, e.g. "18"
  gold_solution    the original GSM8K rationale (with <<calc>> annotations)
  gold_rationale   cleaned rationale used as the assistant's *correct* first answer
  context          = question (kept for compatibility with RankJudge code; never
                   contains the gold answer, so dialogue-only judging stays blind)
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import format_number, parse_number  # noqa: E402


def clean_rationale(solution):
    """Strip calculator annotations and the '#### X' line from a GSM8K solution."""
    body = solution.split("####")[0]
    body = re.sub(r"<<[^>]*>>", "", body)
    lines = [ln.strip() for ln in body.strip().splitlines() if ln.strip()]
    return "\n".join(lines)


def gold_from_solution(solution):
    return format_number(parse_number(solution.split("####")[-1].strip()))


def load_raw(dataset_id, config, split, cache_dir):
    from datasets import load_dataset
    print(f"Loading {dataset_id}/{config} split={split} (cache: {cache_dir})", flush=True)
    return load_dataset(dataset_id, config, split=split, cache_dir=cache_dir)


def preprocess(ds, n, seed, split):
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    records = []
    for i in idxs:
        row = ds[i]
        gold = gold_from_solution(row["answer"])
        if gold is None:
            continue
        records.append({
            "id": f"{split}-{i}",
            "question": row["question"].strip(),
            "gold_answer": gold,
            "gold_solution": row["answer"],
            "gold_rationale": clean_rationale(row["answer"]),
            "context": row["question"].strip(),
        })
        if n and len(records) >= n:
            break
    return records


def main():
    p = argparse.ArgumentParser(description="Download and preprocess GSM8K")
    p.add_argument("--dataset-id", default="openai/gsm8k")
    p.add_argument("--config", default="main")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=300, help="Number of problems to sample (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="../data/raw")
    p.add_argument("--out", default="../data/input/gsm8k.json")
    args = p.parse_args()

    ds = load_raw(args.dataset_id, args.config, args.split, args.cache_dir)
    records = preprocess(ds, args.n, args.seed, args.split)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} GSM8K problems -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

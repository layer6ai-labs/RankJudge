"""Download and preprocess RPC-Bench raw data into the unified context format."""

import argparse
import json
import os
import requests


_RPC_BENCH_URLS = {
    "dev.json": "https://raw.githubusercontent.com/RPC-Bench/PRC-Bench/main/benchmark/dev.json",
    "test.json": "https://raw.githubusercontent.com/RPC-Bench/PRC-Bench/main/benchmark/test.json",
}

_YES_NO_ANSWERS = {"true", "false", "yes", "no"}


def _load_jsonl(path):
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    items = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        pos = next((i for i in range(pos, len(text)) if text[i] in '{['), len(text))
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        items.append(obj)
        pos = end
    return items


def download(raw_dir):
    """Download RPC-Bench papers, combining dev+test into one cached file."""
    os.makedirs(raw_dir, exist_ok=True)
    cached = os.path.join(raw_dir, "rpc_bench.json")
    if os.path.exists(cached):
        print(f"Raw file already exists: {cached}", flush=True)
        return cached
    combined = []
    for name, url in _RPC_BENCH_URLS.items():
        path = os.path.join(raw_dir, name)
        print(f"Downloading {name}...", flush=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
        combined.extend(_load_jsonl(path))
        os.remove(path)
    with open(cached, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined {len(combined)} papers -> {cached}", flush=True)
    return cached


def _filter_qa_pairs(qa_pairs):
    """Remove QA pairs with yes/no/true/false answers."""
    return [qa for qa in qa_pairs if qa["answer"].strip().lower() not in _YES_NO_ANSWERS]


def _format_context(paper, max_qa):
    """Build a single context block from a paper's fields."""
    parts = [f"Title: {paper['title']}"]
    if paper.get("venue"):
        parts.append(f"Venue: {paper['venue']}")
    if paper.get("keywords"):
        parts.append(f"Keywords: {paper['keywords']}")
    parts.append(f"\nAbstract:\n{paper['abstract']}")

    qa_pairs = _filter_qa_pairs(paper.get("qa_pairs", []))
    if max_qa:
        qa_pairs = qa_pairs[:max_qa]
    if qa_pairs:
        qa_lines = "\n".join(
            f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}"
            for i, qa in enumerate(qa_pairs, 1)
        )
        parts.append(f"\nReference QA Pairs:\n{qa_lines}")

    return "\n".join(parts)


def preprocess(raw_path, output_path, max_qa=10):
    """Convert raw RPC-Bench JSON to unified {id, context} format."""
    papers = _load_jsonl(raw_path)
    results = []
    for paper in papers:
        results.append({
            "id": paper["id"],
            "context": _format_context(paper, max_qa),
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Converted {len(results)} RPC-Bench papers -> {output_path}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description="Download and preprocess RPC-Bench dataset")
    p.add_argument("--raw-dir", default="../data/raw",
                   help="Directory for raw downloaded data")
    p.add_argument("--out-dir", default="../data/input",
                   help="Directory for preprocessed output")
    p.add_argument("--output-file", default="ml.json",
                   help="Output JSON filename")
    p.add_argument("--max-qa", type=int, default=10,
                   help="Max QA pairs per paper (after filtering)")
    args = p.parse_args()

    raw_path = download(args.raw_dir)
    output_path = os.path.join(args.out_dir, args.output_file)
    preprocess(raw_path, output_path, args.max_qa)


if __name__ == "__main__":
    main()

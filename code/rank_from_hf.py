"""Mode A: materialize the released Layer6/RankJudge dataset, then rank from it.

Mode A uses the published HF dataset directly. This script:
  1. loads the released `pairs` + `matches` configs (cached under --cache-dir),
  2. writes them as organized JSON into outputs/mode_a/ (pairs.json, matches.json),
  3. computes outputs/mode_a/metrics.json *directly from the local matches.json*.

The released matches are already the published evaluation slice, so no further
filtering is applied (--no-top-removed, top_pct=0).
"""

import argparse, json, os
from datasets import load_dataset
from metrics import compute_metrics


def _row_to_result(row):
    return {
        "id": row["id"],
        "domain": row["domain"],
        "better_is_a": row["better_is_a"],
        "pair": dict(row["pair"]),
        "model": dict(row["model"]),
        "judge": dict(row["judge"]),
        "usage": dict(row["usage"]),
    }


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {len(obj)} records -> {path}")


def materialize(repo_id, split, cache_dir, out_dir):
    """Download (from cache) and dump the released dataset into out_dir."""
    print(f"Loading {repo_id} (split={split}) from cache_dir={cache_dir}")
    ds_pairs = load_dataset(repo_id, "pairs", split=split, cache_dir=cache_dir)
    ds_matches = load_dataset(repo_id, "matches", split=split, cache_dir=cache_dir)
    pairs = [dict(r) for r in ds_pairs]
    matches = [_row_to_result(r) for r in ds_matches]
    _dump(pairs, os.path.join(out_dir, "pairs.json"))
    _dump(matches, os.path.join(out_dir, "matches.json"))
    n_pairs = len({m["id"] for m in matches})
    n_judges = len({m["model"]["name"] for m in matches})
    print(f"  {len(matches)} matches over {n_pairs} pairs x {n_judges} judges")


def parse_args():
    p = argparse.ArgumentParser(description="Materialize the released HF dataset into outputs/mode_a/ and compute metrics")
    p.add_argument("--repo", default="Layer6/RankJudge",
                   help="HF dataset repo id (default: Layer6/RankJudge)")
    p.add_argument("--split", default="train")
    p.add_argument("--cache-dir", default="../data",
                   help="HF datasets cache dir (default: ../data)")
    p.add_argument("--out-dir", default="../outputs/mode_a",
                   help="where to materialize pairs.json / matches.json")
    p.add_argument("--metrics-out", default="../outputs/mode_a/metrics.json")
    p.add_argument("--init-elo", type=int, default=1500)
    args = p.parse_args()
    args.no_top_removed = True
    args.top_pct = 0.0
    return args


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    materialize(args.repo, args.split, args.cache_dir, args.out_dir)

    matches_path = os.path.join(args.out_dir, "matches.json")
    print(f"Computing metrics from {matches_path}")
    with open(matches_path) as f:
        results = json.load(f)
    compute_metrics(results, args)


if __name__ == "__main__":
    main()

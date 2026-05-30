"""Bradley-Terry rating computation + behavior breakdowns.

Per domain, the output is the filtered slice (pairs that are incomplete or
trivial are dropped). In generation mode, a `top_removed` sub-slice is also
included (filtered minus the top X% pairs by BT Elo — the published
evaluation slice). HF-dataset mode skips the top-removed cut via
--no-top-removed since the released matches are already that slice.
"""

import argparse, json
from collections import defaultdict

from utils import collect_matches, run_bradley_terry, split_and_sort, compute_win_rates


def _get_group_val(r, group_key):
    """Get group value from top-level or pair sub-dict."""
    return r.get(group_key, "") or r.get("pair", {}).get(group_key, "")


def _group_pairs_by(results, group_key):
    """Group unique pair IDs by a result field (e.g. assistant_behavior_type)."""
    buckets = defaultdict(set)
    for r in results:
        pid = r["id"]
        val = _get_group_val(r, group_key)
        if val:
            buckets[val].add(pid)
    return buckets


def avg_rating_by(results, pair_ratings, group_key):
    """Average pair rating grouped by a result field (e.g. assistant_behavior_type)."""
    buckets = _group_pairs_by(results, group_key)
    out = {}
    for k in sorted(buckets):
        ratings = [pair_ratings[f"pair:{pid}"] for pid in buckets[k]
                   if pair_ratings.get(f"pair:{pid}") is not None]
        if ratings:
            out[k] = round(sum(ratings) / len(ratings))
    return out


def count_by(results, group_key):
    """Count unique pairs and total matches per group."""
    buckets = _group_pairs_by(results, group_key)
    match_counts = defaultdict(int)
    for r in results:
        val = _get_group_val(r, group_key)
        if val:
            match_counts[val] += 1
    return {
        "pair_count": {k: len(v) for k, v in sorted(buckets.items()) if v},
        "match_count": {k: match_counts[k] for k in sorted(buckets) if match_counts[k]},
    }


def _compute_single(matches, results, args):
    """Compute BT ratings and behavior breakdowns."""
    bt_elo, bt_se = run_bradley_terry(matches, args.init_elo)
    win_rates = compute_win_rates(matches)
    return {
        "BT-Elo": split_and_sort(bt_elo, win_rates, se=bt_se),
        "by_assistant_behavior": {
            "counts": count_by(results, "assistant_behavior_type"),
            "BT-Elo": avg_rating_by(results, bt_elo, "assistant_behavior_type"),
        },
        "by_user_behavior": {
            "counts": count_by(results, "user_behavior_type"),
            "BT-Elo": avg_rating_by(results, bt_elo, "user_behavior_type"),
        },
        "by_domain": {
            "counts": count_by(results, "domain"),
            "BT-Elo": avg_rating_by(results, bt_elo, "domain"),
        },
    }


def _filter_trivial_pairs(matches):
    """Remove pairs that are incomplete (missing any judge) or trivial (all correct / all incorrect)."""
    all_judges = {m[0] for m in matches}
    pair_judges = defaultdict(set)
    pair_results = defaultdict(list)
    for judge, pid, correct in matches:
        pair_judges[pid].add(judge)
        pair_results[pid].append(correct)
    incomplete = {pid for pid, judges in pair_judges.items() if judges != all_judges}
    trivial = {pid for pid, verdicts in pair_results.items()
                if pid not in incomplete and (all(verdicts) or not any(verdicts))}
    excluded = incomplete | trivial
    return [m for m in matches if m[1] not in excluded], len(trivial), len(incomplete)


def _select_top_pids(filtered_matches, args):
    """Pair IDs in the top --top-pct by BT Elo on the `correct` filtered slice."""
    if not filtered_matches:
        return set()
    bt_elo, _ = run_bradley_terry(filtered_matches, args.init_elo)
    pair_elos = {k.split(":", 1)[1]: v for k, v in bt_elo.items() if k.startswith("pair:")}
    if not pair_elos:
        return set()
    sorted_pids = sorted(pair_elos, key=pair_elos.get, reverse=True)
    n_remove = max(1, int(round(len(sorted_pids) * args.top_pct)))
    top = set(sorted_pids[:n_remove])
    print(f"top_removed: dropping top {len(top)} pairs ({args.top_pct:.0%} of {len(sorted_pids)} filtered pairs)")
    return top


def _compute_top_removed(filtered_matches, filtered_results, top_pids, args, correctness_key):
    """Drop the top-ranked pair IDs from filtered_matches and recompute."""
    kept_matches = [m for m in filtered_matches if m[1] not in top_pids]
    kept_pids = {m[1] for m in kept_matches}
    n_removed = len({m[1] for m in filtered_matches}) - len(kept_pids)
    kept_results = [r for r in filtered_results if r.get("id") in kept_pids]

    out = {}
    if kept_matches:
        out.update(_compute_single(kept_matches, kept_results, args))
    out["num_pairs"] = len(kept_pids)
    out["num_top_removed"] = n_removed
    out["top_pct"] = args.top_pct

    print(f"  [{correctness_key}] Top-removed: {len(kept_matches)} matches ({len(kept_pids)} pairs, "
          f"{n_removed} top-ranked pairs removed)")
    return out


def _compute_all(results, args, top_pids):
    matches = collect_matches(results, "correct")
    filtered_matches, n_trivial, n_incomplete = _filter_trivial_pairs(matches)
    filtered_pids = {m[1] for m in filtered_matches}
    filtered_results = [r for r in results if r.get("id") in filtered_pids]

    print(f"  Filtered: {len(filtered_matches)} matches ({len(filtered_pids)} pairs, "
          f"{n_trivial} trivial pairs removed, {n_incomplete} incomplete pairs removed)")

    out = _compute_single(filtered_matches, filtered_results, args)
    out["num_pairs"] = len(filtered_pids)
    out["num_trivial_pairs_removed"] = n_trivial
    out["num_incomplete_pairs_removed"] = n_incomplete

    if not args.no_top_removed:
        out["top_removed"] = _compute_top_removed(
            filtered_matches, filtered_results, top_pids, args, "correct",
        )
    return out


def _compute_for_domains(results, args, top_pids):
    print("--- all ---")
    section = {"all": _compute_all(results, args, top_pids)}
    domains = sorted({r.get("domain", "") for r in results} - {""})
    for domain in domains:
        print(f"--- {domain} ---")
        domain_results = [r for r in results if r.get("domain") == domain]
        section[domain] = _compute_all(domain_results, args, top_pids)
    return section


def _select_global_top_pids(results, args):
    """Top --top-pct pair IDs by BT Elo on the filtered slice."""
    matches = collect_matches(results, "correct")
    filtered_matches, _, _ = _filter_trivial_pairs(matches)
    return _select_top_pids(filtered_matches, args)


def compute_metrics(results, args):
    print(f"Computing metrics from {len(results)} matches...")
    top_pids = set() if args.no_top_removed else _select_global_top_pids(results, args)
    output = _compute_for_domains(results, args, top_pids)
    with open(args.metrics_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nMetrics saved -> {args.metrics_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics from match results")
    p.add_argument("--matches", default="../outputs/mode_b/matches.json")
    p.add_argument("--metrics-out", default="../outputs/mode_b/metrics.json")
    p.add_argument("--init-elo", type=int, default=1500)
    p.add_argument("--top-pct", type=float, default=0.05,
                   help="Fraction of top-Elo pairs to remove for the top_removed slice")
    p.add_argument("--no-top-removed", action="store_true",
                   help="Skip the top_removed slice (use when matches are already top-removed)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.matches) as f:
        results = json.load(f)
    compute_metrics(results, args)


if __name__ == "__main__":
    main()

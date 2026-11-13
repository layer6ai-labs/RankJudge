"""Bradley-Terry rating computation + behavior breakdowns (RankJudge), and
accuracy / CI / baseline metrics for the GSM8K repair experiment (RepairJudge,
see compute_repair_metrics; CLI: python metrics.py --repair --matches a.json b.json).

Per domain, the output is the filtered slice (pairs that are incomplete or
trivial are dropped). In generation mode, a `top_removed` sub-slice is also
included (filtered minus the top X% pairs by BT Elo — the published
evaluation slice). HF-dataset mode skips the top-removed cut via
--no-top-removed since the released matches are already that slice.
"""

import argparse, csv, json, math, os, random
from collections import Counter, defaultdict

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


# =============================== RepairJudge ===============================
#
# Metrics for the GSM8K repair experiment (proposal Sec. 4.4):
#   joint accuracy  = verdict AND flawed round AND error type all correct (RankJudge "correct")
#   winner accuracy = verdict correct
#   round / type accuracy reported separately for diagnosis
#   agree_with_user = how often the judge picked the conversation whose final answer
#                     matches the user's correction (on capitulation pairs this is
#                     the capitulation-reward rate)
# Broken down by condition (dialogue / gold) x judge x pair type, with 95% CIs:
# Wilson intervals for one judge, cluster bootstrap over GSM8K problems when judges
# are pooled. RQ3 deltas (gold - dialogue) use a paired cluster bootstrap.


PAIR_TYPE_LABEL = {"capitulation": "capitulation", "failure_to_revise": "failure_to_revise"}
_ACC_KEYS = {"joint_acc": "correct", "winner_acc": "correct_verdict",
             "round_acc": "correct_bad_round", "type_acc": "correct_behavior_type",
             "agree_with_user": "picked_agree_with_user"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _cluster_of(r):
    return r.get("pair", {}).get("problem_id") or r["id"]


def cluster_bootstrap_ci(records, key, n_boot=2000, seed=0):
    """95% CI for mean(key) resampling GSM8K problems (both pairs + all judges move together)."""
    by_c = defaultdict(list)
    for r in records:
        v = r["judge"].get(key)
        if v is not None:
            by_c[_cluster_of(r)].append(float(v))
    clusters = list(by_c.values())
    if len(clusters) < 2:
        return [None, None]
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        tot = sum(len(c) for c in sample)
        stats.append(sum(sum(c) for c in sample) / tot)
    stats.sort()
    return [round(stats[int(0.025 * n_boot)], 4), round(stats[int(0.975 * n_boot) - 1], 4)]


def _acc_block(records, pooled):
    out = {"n": len(records),
           "n_parse_fail": sum(1 for r in records if r["judge"].get("correct") is None)}
    for name, key in _ACC_KEYS.items():
        vals = [r["judge"].get(key) for r in records if r["judge"].get(key) is not None]
        if not vals:
            out[name] = None
            continue
        k, n = sum(bool(v) for v in vals), len(vals)
        ci = cluster_bootstrap_ci(records, key) if pooled else wilson_ci(k, n)
        out[name] = {"acc": round(k / n, 4), "k": k, "n": n, "ci95": ci}
    out["pred_type_dist"] = dict(Counter(r["judge"].get("behavior_type_pred") for r in records
                                         if r["judge"].get("behavior_type_pred")).most_common())
    return out


def _subsets(records):
    yield "all", records
    for pt, label in PAIR_TYPE_LABEL.items():
        yield label, [r for r in records if r.get("pair", {}).get("pair_type") == pt]


def repair_baselines(records):
    """Winner accuracy of heuristic baselines, computed once per unique pair."""
    pairs = {}
    for r in records:
        if r.get("better_is_a") is not None:
            pairs[r["id"]] = (r["better_is_a"], r["pair"].get("user_agree_is_a"), r["pair"].get("pair_type"))
    out = {}
    for label, sel in [("all", None)] + [(v, k) for k, v in PAIR_TYPE_LABEL.items()]:
        ps = [p for p in pairs.values() if sel is None or p[2] == sel]
        n = len(ps)
        pos = sum(1 for b, _, _ in ps if b)                          # always pick A
        agree = sum(1 for b, u, _ in ps if u is not None and u == b)  # pick the convo agreeing with the user
        out[label] = {
            "n_pairs": n,
            "position_always_A": {"winner_acc": round(pos / n, 4) if n else None, "ci95": wilson_ci(pos, n)},
            "agree_with_user": {"winner_acc": round(agree / n, 4) if n else None, "ci95": wilson_ci(agree, n)},
        }
    return out


def paired_delta(gold_recs, dlg_recs, key, n_boot=2000, seed=0):
    """RQ3: acc(gold) - acc(dialogue) on (pair, judge) items present in both, paired cluster bootstrap."""
    g = {(r["id"], r["model"]["name"]): r for r in gold_recs if r["judge"].get(key) is not None}
    d = {(r["id"], r["model"]["name"]): r for r in dlg_recs if r["judge"].get(key) is not None}
    common = sorted(set(g) & set(d))
    if not common:
        return None
    by_c = defaultdict(list)
    for k in common:
        by_c[_cluster_of(g[k])].append(float(g[k]["judge"][key]) - float(d[k]["judge"][key]))
    clusters = list(by_c.values())
    n = len(common)
    delta = sum(sum(c) for c in clusters) / n
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        stats.append(sum(sum(c) for c in sample) / sum(len(c) for c in sample))
    stats.sort()
    return {"delta": round(delta, 4), "n": n,
            "ci95": [round(stats[int(0.025 * n_boot)], 4), round(stats[int(0.975 * n_boot) - 1], 4)]}


def compute_repair_metrics(results, metrics_out, table_out=None):
    results = [r for r in results if r.get("domain") == "gsm8k"]
    conditions = sorted({r.get("condition", "dialogue") for r in results})
    judges = sorted({r["model"]["name"] for r in results})
    out = {"baselines": repair_baselines(results), "conditions": {}, "rq3_gold_minus_dialogue": {}}
    rows = []
    for cond in conditions:
        cr = [r for r in results if r.get("condition", "dialogue") == cond]
        block = {}
        for judge in ["ALL_JUDGES"] + judges:
            jr = cr if judge == "ALL_JUDGES" else [r for r in cr if r["model"]["name"] == judge]
            block[judge] = {}
            for label, sub in _subsets(jr):
                m = _acc_block(sub, pooled=(judge == "ALL_JUDGES"))
                block[judge][label] = m
                row = {"condition": cond, "judge": judge, "pair_type": label, "n": m["n"]}
                for name in _ACC_KEYS:
                    if m[name]:
                        row[name] = m[name]["acc"]
                        row[name + "_lo"], row[name + "_hi"] = m[name]["ci95"]
                rows.append(row)
        out["conditions"][cond] = block

    if "gold" in conditions and "dialogue" in conditions:
        g_all = [r for r in results if r.get("condition") == "gold"]
        d_all = [r for r in results if r.get("condition") == "dialogue"]
        for judge in ["ALL_JUDGES"] + judges:
            out["rq3_gold_minus_dialogue"][judge] = {}
            for label, g_sub in _subsets(g_all if judge == "ALL_JUDGES" else [r for r in g_all if r["model"]["name"] == judge]):
                d_sub = dict(_subsets(d_all if judge == "ALL_JUDGES" else [r for r in d_all if r["model"]["name"] == judge]))[label]
                out["rq3_gold_minus_dialogue"][judge][label] = {
                    "joint_acc": paired_delta(g_sub, d_sub, "correct"),
                    "winner_acc": paired_delta(g_sub, d_sub, "correct_verdict"),
                }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_out)), exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(out, f, indent=2)
    table_out = table_out or os.path.splitext(metrics_out)[0] + "_table.csv"
    if rows:
        fields = ["condition", "judge", "pair_type", "n"] + [
            k for name in _ACC_KEYS for k in (name, name + "_lo", name + "_hi")]
        with open(table_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    _print_repair_summary(out)
    print(f"\nRepair metrics saved -> {metrics_out}\nFlat table -> {table_out}")
    return out


def _fmt(m, key):
    v = m.get(key)
    if not v:
        return "-"
    lo, hi = v["ci95"]
    ci = "" if lo is None else f" [{lo:.2f},{hi:.2f}]"
    return f"{v['acc']:.2f}{ci}"


def _print_repair_summary(out):
    b = out["baselines"]
    print("\nBaselines (winner accuracy):")
    for label, v in b.items():
        print(f"  {label:18s} n={v['n_pairs']:4d}  always-A={v['position_always_A']['winner_acc']}  "
              f"agree-with-user={v['agree_with_user']['winner_acc']}")
    for cond, block in out["conditions"].items():
        print(f"\n== condition: {cond} ==")
        print(f"  {'judge':20s} {'pair type':18s} {'n':>4s}  {'joint':18s} {'winner':18s} agree-w-user")
        for judge, subs in block.items():
            for label, m in subs.items():
                print(f"  {judge:20s} {label:18s} {m['n']:4d}  {_fmt(m, 'joint_acc'):18s} "
                      f"{_fmt(m, 'winner_acc'):18s} {_fmt(m, 'agree_with_user')}")
    if out["rq3_gold_minus_dialogue"]:
        print("\n== RQ3: gold - dialogue (joint acc) ==")
        for judge, subs in out["rq3_gold_minus_dialogue"].items():
            parts = [f"{label}={v['joint_acc']['delta']:+.3f} [{v['joint_acc']['ci95'][0]:+.2f},{v['joint_acc']['ci95'][1]:+.2f}]"
                     for label, v in subs.items() if v["joint_acc"]]
            print(f"  {judge:20s} " + "  ".join(parts))


def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics from match results")
    p.add_argument("--matches", nargs="+", default=["../outputs/repair/matches_dialogue.json", "../outputs/repair/matches_gold.json"],
                   help="Matches files: pass both the dialogue and gold files")
    p.add_argument("--repair", action="store_true",
                   help="Our metrics (joint/winner acc, CIs, baselines, RQ3 deltas). Without it: RankJudge BT-Elo (Mode A)")
    p.add_argument("--metrics-out", default="../outputs/repair/metrics.json")
    p.add_argument("--init-elo", type=int, default=1500)
    p.add_argument("--top-pct", type=float, default=0.05,
                   help="Fraction of top-Elo pairs to remove for the top_removed slice")
    p.add_argument("--no-top-removed", action="store_true",
                   help="Skip the top_removed slice (use when matches are already top-removed)")
    return p.parse_args()


def main():
    args = parse_args()
    results = []
    for path in args.matches:
        with open(path) as f:
            results.extend(json.load(f))
    if args.repair:
        compute_repair_metrics(results, args.metrics_out)
    else:
        compute_metrics(results, args)


if __name__ == "__main__":
    main()

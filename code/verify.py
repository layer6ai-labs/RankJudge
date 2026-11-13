"""Verify GSM8K user-correction pairs.

Per pair: free numeric + structural checks against the GSM8K gold answer
(repair_rule_checks), then one LLM audit call (repair_llm_check) if those pass.
Only pairs that pass every check go to pairs_filtered.json.
"""

import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import call_structured, get_api_key, resolve_api_url, extract_final_number, numbers_equal
from prompts import REPAIR_CHECK
from schemas import REPAIR_CHECK_SCHEMA


_GOALS = {
    "capitulation": ("keep the correct first answer and explain why the user's value is wrong",
                         "capitulate: abandon the correct answer and adopt the user's wrong value"),
    "failure_to_revise": ("acknowledge the error and revise to the user's (correct) value",
                         "fail to revise: defend the original wrong answer"),
}


def _good_bad(pair):
    return ((pair["convo_a"], pair["convo_b"]) if pair["better_is_a"]
            else (pair["convo_b"], pair["convo_a"]))


def _words(text):
    return len(text.split())


def repair_rule_checks(pair, min_length_ratio):
    """Deterministic checks that need no API call. Returns {check_name: bool}."""
    m = pair["metadata"]
    good, bad = _good_bad(pair)
    gold, w = m["gold_answer"], m["wrong_answer"]
    exp_first = gold if m["pair_type"] == "capitulation" else w
    exp_user = w if m["pair_type"] == "capitulation" else gold
    g_len, b_len = _words(good[-1]["content"]), _words(bad[-1]["content"])
    return {
        "four_turns": len(good) == 4 and len(bad) == 4,
        "shared_prefix": good[:3] == bad[:3],
        "gold_ne_wrong": not numbers_equal(gold, w),
        "first_answer_value": numbers_equal(extract_final_number(good[1]["content"]), exp_first),
        "user_correction_value": numbers_equal(extract_final_number(good[2]["content"]), exp_user),
        "good_final_is_gold": numbers_equal(extract_final_number(good[3]["content"]), m["good_final_value"]),
        "bad_final_is_flawed_value": numbers_equal(extract_final_number(bad[3]["content"]), m["bad_final_value"]),
        "length_matched": min(g_len, b_len) / max(g_len, b_len, 1) >= min_length_ratio,
    }


def repair_llm_check(pair, model_id, api_key, api_url, max_tokens):
    """One LLM audit call: final answers, stance, reasoning validity, extra flaws."""
    m = pair["metadata"]
    good, bad = _good_bad(pair)
    good_goal, bad_goal = _GOALS[m["pair_type"]]
    prompt = REPAIR_CHECK.format(
        good_goal=good_goal, bad_goal=bad_goal, question=m["context"], gold=m["gold_answer"],
        convo=format_convo_flat(good[:3]), good=good[3]["content"], bad=bad[3]["content"],
        first=m["first_answer_value"], user_value=m["user_correction_value"])
    result, _ = call_structured(model_id, prompt, api_key, api_url, REPAIR_CHECK_SCHEMA, max_tokens)
    keeps_good = "keeps_original" if m["pair_type"] == "capitulation" else "adopts_user"
    keeps_bad = "adopts_user" if m["pair_type"] == "capitulation" else "keeps_original"
    checks = {
        "llm_good_answer": numbers_equal(result["good_final_answer"], m["good_final_value"]),
        "llm_bad_answer": numbers_equal(result["bad_final_answer"], m["bad_final_value"]),
        "llm_good_stance": result["good_stance"] == keeps_good,
        "llm_bad_stance": result["bad_stance"] == keeps_bad,
        "llm_good_reasoning_valid": bool(result["good_reasoning_valid"]),
        "llm_bad_single_flaw": not result["bad_other_flaws"],
    }
    return checks, result


def format_convo_flat(convo):
    return "\n\n".join(f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}" for t in convo)


def verify_repair(pair, model_id, api_key, api_url, max_tokens, use_llm=True, min_length_ratio=0.6):
    checks = repair_rule_checks(pair, min_length_ratio)
    llm = None
    # Skip the paid call if the free checks already fail.
    if use_llm and all(checks.values()):
        llm_checks, llm = repair_llm_check(pair, model_id, api_key, api_url, max_tokens)
        checks.update(llm_checks)
    failed = [k for k, v in checks.items() if not v]
    return {"id": pair["id"], "domain": "gsm8k", "kind": "repair",
            "pair_type": pair["metadata"]["pair_type"],
            "checks": checks, "llm": llm, "failed": failed,
            "passed": not failed and (llm is not None or not use_llm)}


def verify_pair(pair, model_id, api_key, api_url, max_tokens, args):
    return verify_repair(pair, model_id, api_key, api_url, max_tokens,
                         args.repair_llm_check, args.min_length_ratio)


def load_state(args):
    results, done_ids = [], set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} done, {len(results)} verifications loaded.")
    return results, done_ids


def save_results(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def pair_passes(v):
    return v["passed"]


def filter_pairs(pairs, results):
    by_id = {r["id"]: r for r in results}
    return [p for p in pairs if p["id"] in by_id and pair_passes(by_id[p["id"]])]


def save_filtered(pairs, results, args):
    kept = filter_pairs(pairs, results)
    with open(args.filtered_out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"Filtered: {len(kept)}/{len(pairs)} pairs passed -> {args.filtered_out}")
    by_type = {}
    for p in pairs:
        t = p.get("metadata", {}).get("pair_type")
        if t:
            by_type.setdefault(t, [0, 0])[1] += 1
    for p in kept:
        t = p.get("metadata", {}).get("pair_type")
        if t:
            by_type[t][0] += 1
    for t, (k, n) in sorted(by_type.items()):
        print(f"  {t}: {k}/{n} passed")


def run_verification(pairs, results, done_ids, args):
    pairs = [p for p in pairs if p["id"] not in done_ids]
    print(f"Verifying {len(pairs)} pairs with {args.model}...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify_pair, p, args.model, args.api_key,
                          args.api_url, args.max_tokens, args): p for p in pairs}
        for i, fut in enumerate(as_completed(futs), 1):
            pair = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(pairs)}] {pair['id']} ({r['pair_type']}) - "
                      f"{'PASS' if r['passed'] else 'FAIL ' + ','.join(r['failed'])}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(pairs)}] {pair['id']} - FAILED: {e}", flush=True)
            if i % args.save_interval == 0:
                save_results(args.out, results)
                print(f"Saved -> {args.out}", flush=True)
    save_results(args.out, results)
    print(f"Done. {len(results)} total verifications -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Verify GSM8K user-correction pairs")
    p.add_argument("--pairs", default="../outputs/repair/pairs.json")
    p.add_argument("--out", default="../outputs/repair/verification.json")
    p.add_argument("--filtered-out", default="../outputs/repair/pairs_filtered.json")
    p.add_argument("--api-key-path", default="../api_key.json")
    p.add_argument("--backend", default="local",
                   help="'local' (model server at LOCAL_API_URL, default) or a full chat-completions URL")
    p.add_argument("--model", default="gpt-oss:120b",
                   help="Model name on the server, e.g. gpt-oss:120b")
    p.add_argument("--max-tokens", type=int, default=1024 * 32)
    p.add_argument("--workers", type=int, default=2, help="Parallel requests (16 is fine with vLLM)")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--repair-llm-check", action=argparse.BooleanOptionalAction, default=True,
                   help="Also run the LLM audit (default). --no-repair-llm-check = free rule checks only.")
    p.add_argument("--min-length-ratio", type=float, default=0.6,
                   help="Min word-count ratio between good and bad final turns")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pairs) as f:
        pairs = json.load(f)
    args.api_url = resolve_api_url(args.backend)
    args.api_key = get_api_key(args.api_key_path, [args.api_url]) if args.repair_llm_check else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results, done_ids = load_state(args)
    run_verification(pairs, results, done_ids, args)
    save_filtered(pairs, results, args)


if __name__ == "__main__":
    main()

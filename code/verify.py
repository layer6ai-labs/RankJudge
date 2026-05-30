"""Three-layer verification for synthetic conversation pairs: coherence, adherence, grounding."""

import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import call_structured, load_api_key
from prompts import COHERENCE, ADHERENCE, GROUNDING
from schemas import COHERENCE_SCHEMA, ADHERENCE_SCHEMA, GROUNDING_SCHEMA
from taxonomy import ASSISTANT_BEHAVIORS, USER_BEHAVIORS


def format_convo(convo):
    """Render a conversation as numbered rounds for prompts."""
    lines = []
    for i in range(0, len(convo), 2):
        r = i // 2 + 1
        lines.append(f"[Round {r}] USER: {convo[i]['content']}")
        lines.append(f"[Round {r}] ASSISTANT: {convo[i+1]['content']}")
    return "\n".join(lines)


def format_assistant_turns(convo):
    """Render only assistant turns with 1-indexed round numbers."""
    lines = []
    for i in range(1, len(convo), 2):
        r = i // 2 + 1
        lines.append(f"[Round {r}] ASSISTANT: {convo[i]['content']}")
    return "\n".join(lines)


def verify_coherence(pair, model_id, api_key, api_url, max_tokens):
    m = pair["metadata"]
    beh = m["assistant_behavior_type"]
    prompt = COHERENCE.format(
        context=m["context"],
        user_behavior_name=m["user_behavior_type"],
        user_behavior_desc=USER_BEHAVIORS[m["user_behavior_type"]],
        assistant_behavior_name=beh,
        virtue=ASSISTANT_BEHAVIORS[beh]["virtue"],
        flaw=ASSISTANT_BEHAVIORS[beh]["flaw"],
        n_rounds=m["n_rounds"],
        bad_round_index=m["bad_round_index"],
        good_plan=pair["plan"]["good"],
        bad_plan=pair["plan"]["bad"])
    result, _ = call_structured(model_id, prompt, api_key, api_url, COHERENCE_SCHEMA, max_tokens)
    return result


def verify_adherence(pair, model_id, api_key, api_url, max_tokens):
    m = pair["metadata"]
    beh = m["assistant_behavior_type"]
    good_convo, bad_convo = ((pair["convo_a"], pair["convo_b"]) if pair["better_is_a"]
                             else (pair["convo_b"], pair["convo_a"]))
    prompt = ADHERENCE.format(
        user_behavior_name=m["user_behavior_type"],
        user_behavior_desc=USER_BEHAVIORS[m["user_behavior_type"]],
        assistant_behavior_name=beh,
        virtue=ASSISTANT_BEHAVIORS[beh]["virtue"],
        flaw=ASSISTANT_BEHAVIORS[beh]["flaw"],
        n_rounds=m["n_rounds"],
        bad_round_index=m["bad_round_index"],
        good_plan=pair["plan"]["good"],
        bad_plan=pair["plan"]["bad"],
        good_convo=format_convo(good_convo),
        bad_convo=format_convo(bad_convo))
    result, _ = call_structured(model_id, prompt, api_key, api_url, ADHERENCE_SCHEMA, max_tokens)
    return result


def _ground_one(convo, label, skip_rounds, context, model_id, api_key, api_url, max_tokens):
    prompt = GROUNDING.format(
        label=label,
        context=context,
        skip_rounds=skip_rounds,
        turns=format_assistant_turns(convo))
    result, _ = call_structured(model_id, prompt, api_key, api_url, GROUNDING_SCHEMA, max_tokens)
    rounds = [r for r in result.get("rounds", []) if r["round_index"] not in skip_rounds]
    total = sum(len(r["claims"]) for r in rounds)
    grounded = sum(c["grounded"] for r in rounds for c in r["claims"])
    rate = grounded / total if total else None
    return {"rounds": rounds, "grounded_rate": rate}


def verify_grounding(pair, model_id, api_key, api_url, max_tokens):
    m = pair["metadata"]
    good_convo, bad_convo = ((pair["convo_a"], pair["convo_b"]) if pair["better_is_a"]
                             else (pair["convo_b"], pair["convo_a"]))
    good = _ground_one(good_convo, "good", [], m["context"], model_id, api_key, api_url, max_tokens)
    bad = _ground_one(bad_convo, "bad", [m["bad_round_index"]], m["context"], model_id, api_key, api_url, max_tokens)
    return {"good": good, "bad": bad}


def verify_pair(pair, model_id, api_key, api_url, max_tokens):
    pid = pair["id"]
    try:
        coherence = verify_coherence(pair, model_id, api_key, api_url, max_tokens)
        adherence = verify_adherence(pair, model_id, api_key, api_url, max_tokens)
        grounding = verify_grounding(pair, model_id, api_key, api_url, max_tokens)
    except Exception as e:
        print(f"  [{pid}] ERROR: {e}", flush=True)
        raise
    return {"id": pid, "coherence": coherence, "adherence": adherence, "grounding": grounding}


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
    c, a, g = v["coherence"], v["adherence"], v["grounding"]
    return (c["good_ok"] and c["bad_ok"]
            and a["good_followed"] and a["bad_followed"] and a["bad_flaw_round_correct"]
            and g["good"]["grounded_rate"] == 1.0 and g["bad"]["grounded_rate"] == 1.0)


def filter_pairs(pairs, results):
    by_id = {r["id"]: r for r in results}
    return [p for p in pairs if p["id"] in by_id and pair_passes(by_id[p["id"]])]


def save_filtered(pairs, results, args):
    kept = filter_pairs(pairs, results)
    with open(args.filtered_out, "w") as f:
        json.dump(kept, f, indent=2)
    print(f"Filtered: {len(kept)}/{len(pairs)} pairs passed -> {args.filtered_out}")


def run_verification(pairs, results, done_ids, args):
    pairs = [p for p in pairs if p["id"] not in done_ids]
    print(f"Verifying {len(pairs)} pairs with {args.model}...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(verify_pair, p, args.model, args.api_key,
                          args.api_url, args.max_tokens): p for p in pairs}
        for i, fut in enumerate(as_completed(futs), 1):
            pair = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                g = r["grounding"]["good"]["grounded_rate"]
                b = r["grounding"]["bad"]["grounded_rate"]
                print(f"[{i}/{len(pairs)}] {pair['id']} - coh(g={r['coherence']['good_ok']},b={r['coherence']['bad_ok']}) "
                      f"adh(g={r['adherence']['good_followed']},b={r['adherence']['bad_flaw_round_correct']}) "
                      f"ground(g={g},b={b})", flush=True)
            except Exception as e:
                print(f"[{i}/{len(pairs)}] {pair['id']} - FAILED: {e}", flush=True)
            if i % args.save_interval == 0:
                save_results(args.out, results)
                print(f"Saved -> {args.out}", flush=True)
    save_results(args.out, results)
    print(f"Done. {len(results)} total verifications -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Three-layer verification for synthetic pairs")
    p.add_argument("--pairs", default="../outputs/mode_b/pairs.json")
    p.add_argument("--out", default="../outputs/mode_b/verification.json")
    p.add_argument("--filtered-out", default="../outputs/mode_b/pairs_filtered.json")
    p.add_argument("--api-key-path", default="../api_key.json")
    p.add_argument("--api-url", default="https://openrouter.ai/api/v1/chat/completions")
    p.add_argument("--model", default="openai/gpt-5.5")
    p.add_argument("--max-tokens", type=int, default=1024 * 32)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main():
    args = parse_args()
    args.api_key = load_api_key(args.api_key_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.pairs) as f:
        pairs = json.load(f)
    results, done_ids = load_state(args)
    run_verification(pairs, results, done_ids, args)
    save_filtered(pairs, results, args)


if __name__ == "__main__":
    main()

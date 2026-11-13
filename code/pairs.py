"""Generate GSM8K user-correction conversation pairs.

Each pair is a 2-round / 4-turn conversation:
  turn 1  USER       the GSM8K question                                   (template)
  turn 2  ASSISTANT  first answer: gold rationale OR a generated wrong one  (template / 1 LLM call per problem)
  turn 3  USER       repair, e.g. "Shouldn't it be X?"                      (template)
  turn 4  ASSISTANT  good vs. bad final response                          (1 LLM call per pair)
Turns 1-3 are identical inside a pair; only turn 4 (= round 2) differs.
"""

import argparse, hashlib, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (call_structured, get_api_key, resolve_api_url, load_gsm8k, seed_everything,
                   extract_final_number, format_number, numbers_equal, parse_number)
from prompts import WRONG_SOLUTION, FINAL_TURN_WRONG_CORRECTION, FINAL_TURN_RIGHT_CORRECTION
from taxonomy import PAIR_TYPE_TO_BEHAVIOR
from schemas import WRONG_SOLUTION_SCHEMA, FINAL_TURN_SCHEMA
from sampler import sample_repair_params


def make_id(domain, raw_id):
    return hashlib.sha256(f"{domain}:{raw_id}".encode()).hexdigest()[:12]


def load_state(args):
    results, done_ids = [], set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} done, {len(results)} pairs loaded.")
    return results, done_ids


def save_pairs(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _format_convo_text(convo):
    return "\n\n".join(f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['content']}" for t in convo)


def correct_first_answer(problem):
    """Turn 2 when the assistant is right: the cleaned GSM8K rationale + a fixed closing line."""
    return f"{problem['gold_rationale']}\nThe answer is {problem['gold_answer']}."


def wrong_value_candidates(problem):
    """Plausible wrong answers W, most natural first. Chosen by code, not by the model.

    1. intermediate results of the gold solution, last first (= "stopped a step early"
       or "used the wrong intermediate value");
    2. generic slips: x2, /2, +1, -1 (+2, +10 as a last resort).
    Values differ from the gold answer, are positive when the gold is, and are integers
    when the gold is.
    """
    gold = parse_number(problem["gold_answer"])
    inter = [parse_number(x) for x in re.findall(r"<<[^>]*?=([^>]+)>>", problem.get("gold_solution", ""))]
    generic = [gold * 2, gold / 2, gold + 1, gold - 1, gold + 2, gold + 10]
    out = []
    for v in list(reversed(inter)) + generic:
        if v is None or (gold > 0 and v <= 0) or numbers_equal(v, gold):
            continue
        if gold == gold.to_integral_value() and v != v.to_integral_value():
            continue
        f = format_number(v)
        if f not in out:
            out.append(f)
    return out


def generate_wrong_solution(problem, model_id, api_key, api_url, max_tokens, max_retries=3):
    """Turn 2 when the assistant is wrong: code picks W, the model writes a solution reaching W.

    The same W is reused as the user's (incorrect) correction in the capitulation
    pair, so both pairs from one problem share {gold, W}. If the model cannot reach a W
    after two tries, the next candidate is used.
    """
    gold = problem["gold_answer"]
    cands = wrong_value_candidates(problem)
    if not cands:
        raise ValueError("no plausible wrong value")
    last_err = None
    for attempt in range(max_retries):
        wrong = cands[min(attempt // 2, len(cands) - 1)]
        prompt = WRONG_SOLUTION.format(question=problem["question"], gold=gold, wrong=wrong,
                                       gold_rationale=problem["gold_rationale"])
        result, _ = call_structured(model_id, prompt, api_key, api_url, WRONG_SOLUTION_SCHEMA, max_tokens)
        solution = result["solution"].strip()
        if not numbers_equal(extract_final_number(solution), wrong):
            last_err = f"solution ends with {extract_final_number(solution)}, expected {wrong}"
        else:
            return {"solution": solution, "wrong_answer": wrong, "mistake": result["mistake"],
                    "wrong_candidates": cands}
        print(f"  [{problem['id']}] wrong-solution check failed (attempt {attempt + 1}/{max_retries}): {last_err}", flush=True)
    raise ValueError(f"wrong solution failed: {last_err}")


def build_prefix(problem, pair_type, wrong, repair):
    """Turns 1-3 (shared by both conversations in a pair) plus the bookkeeping values."""
    gold, w = problem["gold_answer"], wrong["wrong_answer"]
    if pair_type == "capitulation":      # assistant right, user wrong
        first_text, first_val, user_val = correct_first_answer(problem), gold, w
    else:                                    # failure_to_revise: assistant wrong, user right
        first_text, first_val, user_val = wrong["solution"], w, gold
    prefix = [
        {"role": "user", "content": problem["question"]},
        {"role": "assistant", "content": first_text},
        {"role": "user", "content": repair["repair_template"].format(x=user_val)},
    ]
    return prefix, first_val, user_val


def generate_final_turns(problem, pair_type, prefix, first_val, user_val,
                         model_id, api_key, api_url, max_tokens):
    """Turn 4: one LLM call returns both the good and the bad final response."""
    template = FINAL_TURN_WRONG_CORRECTION if pair_type == "capitulation" else FINAL_TURN_RIGHT_CORRECTION
    prompt = template.format(question=problem["question"], gold=problem["gold_answer"],
                             convo=_format_convo_text(prefix), first=first_val, user_value=user_val)
    result, _ = call_structured(model_id, prompt, api_key, api_url, FINAL_TURN_SCHEMA, max_tokens)
    return result["good_response"].strip(), result["bad_response"].strip(), result.get("plan", "")


def make_gsm8k_pair(problem, pair_type, wrong, model_id, api_key, api_url, max_tokens, max_retries=3):
    behavior = PAIR_TYPE_TO_BEHAVIOR[pair_type]
    repair = sample_repair_params()
    prefix, first_val, user_val = build_prefix(problem, pair_type, wrong, repair)
    gold = problem["gold_answer"]
    # Value the bad final turn should end on: the user's wrong value (capitulation)
    # or the assistant's own wrong first answer (failure to revise). Either way = W.
    bad_val = user_val if pair_type == "capitulation" else first_val
    for attempt in range(1, max_retries + 1):
        good, bad, plan = generate_final_turns(problem, pair_type, prefix, first_val, user_val,
                                               model_id, api_key, api_url, max_tokens)
        g, b = extract_final_number(good), extract_final_number(bad)
        if numbers_equal(g, gold) and numbers_equal(b, bad_val):
            break
        print(f"  [{problem['id']}] {pair_type}: final turns end on good={g}, bad={b} "
              f"(want {gold}, {bad_val}); attempt {attempt}/{max_retries}", flush=True)
    # A pair that still ends on the wrong numbers is kept here and dropped by verify.py.

    good_convo = prefix + [{"role": "assistant", "content": good}]
    bad_convo = prefix + [{"role": "assistant", "content": bad}]
    better_is_a = random.random() < 0.5
    convo_a, convo_b = (good_convo, bad_convo) if better_is_a else (bad_convo, good_convo)
    # The conversation whose final answer matches the user's correction ("agree-with-user"):
    # bad convo for capitulation, good convo for failure_to_revise.
    agree_is_good = pair_type == "failure_to_revise"
    user_agree_is_a = better_is_a if agree_is_good else not better_is_a

    return {
        "id": make_id("gsm8k", f"{problem['id']}:{pair_type}"),
        "domain": "gsm8k",
        "convo_a": convo_a, "convo_b": convo_b,
        "better_is_a": better_is_a,
        "plan": {"good": plan, "bad": plan},   # one generator call wrote both final turns
        "metadata": {
            "problem_id": problem["id"],
            "pair_type": pair_type,
            "assistant_behavior_type": behavior,
            "user_behavior_type": f"repair:{repair['repair_strategy']}",
            "repair_strategy": repair["repair_strategy"],
            "repair_template": repair["repair_template"],
            "n_rounds": 2,
            "bad_round_index": 2,          # round 2 = turn 4 (the final assistant message)
            "gold_answer": gold,
            "wrong_answer": wrong["wrong_answer"],
            "first_answer_value": first_val,
            "user_correction_value": user_val,
            "good_final_value": gold,
            "bad_final_value": bad_val,
            "user_agree_is_a": user_agree_is_a,
            "wrong_solution_mistake": wrong["mistake"],
            "context": problem["question"],   # never contains the gold answer
        },
    }


def process_problem(problem, model_id, api_key, api_url, max_tokens, pair_types):
    """One problem -> up to two pairs (capitulation, failure_to_revise)."""
    wrong = generate_wrong_solution(problem, model_id, api_key, api_url, max_tokens)
    out = []
    for pt in pair_types:
        try:
            out.append(make_gsm8k_pair(problem, pt, wrong, model_id, api_key, api_url, max_tokens))
        except Exception as e:
            print(f"  [{problem['id']}] {pt} - ERROR: {e}", flush=True)
    return out


def run_generation_gsm8k(problems, results, done_ids, args):
    def todo(p):
        return [pt for pt in args.pair_types if make_id("gsm8k", f"{p['id']}:{pt}") not in done_ids]
    jobs = [(p, todo(p)) for p in problems]
    jobs = [(p, pts) for p, pts in jobs if pts]
    print(f"Processing {len(jobs)} gsm8k problems with {args.model}...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_problem, p, args.model, args.api_key, args.api_url,
                          args.max_tokens, pts): p for p, pts in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            problem = futs[fut]
            try:
                new = fut.result()
                results.extend(new)
                print(f"[{i}/{len(jobs)}] {problem['id']} - {len(new)} pairs ({len(results)} total)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {problem['id']} - ERROR: {e}", flush=True)
            if i % args.save_interval == 0:
                save_pairs(args.out, results)
                print(f"Saved -> {args.out}", flush=True)
    save_pairs(args.out, results)
    print(f"Done gsm8k. {len(results)} total pairs -> {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate GSM8K user-correction pairs")
    p.add_argument("--data-dir", default="../data", help="Directory holding input/gsm8k.json")
    p.add_argument("--out", default="../outputs/repair/pairs.json")
    p.add_argument("--api-key-path", default="../api_key.json")
    p.add_argument("--backend", default="local",
                   help="'local' (model server at LOCAL_API_URL, default) or a full chat-completions URL")
    p.add_argument("--model", default="gpt-oss:120b",
                   help="Model name on the server, e.g. gpt-oss:120b")
    p.add_argument("--max-tokens", type=int, default=1024 * 32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2, help="Parallel requests (16 is fine with vLLM)")
    p.add_argument("--n-samples", type=int, default=10, help="Number of GSM8K problems to use")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pair-types", nargs="*", default=["capitulation", "failure_to_revise"],
                   choices=["capitulation", "failure_to_revise"],
                   help="Which pair types to build per problem (default: both)")
    return p.parse_args()


def main():
    args = parse_args()
    args.api_url = resolve_api_url(args.backend)
    args.api_key = get_api_key(args.api_key_path, [args.api_url])
    print(f"Generator: {args.model} @ {args.api_url}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results, done_ids = load_state(args)
    seed_everything(args.seed)
    problems = load_gsm8k(args.data_dir, args.n_samples)
    run_generation_gsm8k(problems, results, done_ids, args)


if __name__ == "__main__":
    main()

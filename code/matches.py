"""
RankJudge match runner (multi-turn).

Judge LLMs evaluate pairs of multi-turn conversations.
Both judges and pairs get Elo ratings.
A judge "wins" if it correctly identifies the better conversation.
"""

import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import call_openrouter, load_api_key
from metrics import compute_metrics
from prompts import JUDGE, LETTER_TO_BEHAVIOR
from models import MODELS
from schemas import JUDGE_SCHEMA


def format_convo(convo):
    lines = []
    for turn in convo:
        label = "USER" if turn["role"] == "user" else "ASSISTANT"
        lines.append(f"{label}: {turn['content']}")
    return "\n\n".join(lines)


def build_judge_prompt(pair, with_context=False):
    text_a = format_convo(pair["convo_a"])
    text_b = format_convo(pair["convo_b"])
    meta = pair.get("metadata", {})
    context_section = ""
    if with_context:
        context = meta.get("context", "")
        context_section = f"\n=== Reference Context ===\n{context}\n"
    return JUDGE.format(context_section=context_section, text_a=text_a, text_b=text_b)


JUDGE_RESP_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "response", "strict": True, "schema": JUDGE_SCHEMA}
}


# --- State management ---

def load_state(args):
    results, done_keys = [], set()
    if args.resume and os.path.exists(args.matches_out):
        with open(args.matches_out) as f:
            results = json.load(f)
        done_keys = {(r["id"], r["model"]["name"]) for r in results if r.get("judge", {}).get("answer") is not None}
        results = [r for r in results if (r["id"], r["model"]["name"]) in done_keys]
        print(f"Resuming: {len(done_keys)} done, {len(results)} matches loaded.")
    return results, done_keys


def save_matches(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# --- Job building and execution ---

def build_judge_configs(models):
    configs = []
    for m in models:
        configs.append((m["judge_name"], m["openrouter_name"], m["add_context"], m["support_thinking"], m.get("use_thinking", m["support_thinking"])))
    return configs


def build_jobs(pairs, judge_configs, done_keys):
    jobs = []
    for pair in pairs:
        pid = pair["id"]
        for name, model_id, with_context, support_thinking, use_thinking in judge_configs:
            if (pid, name) not in done_keys:
                jobs.append((pair, name, model_id, with_context, support_thinking, use_thinking))
    return jobs


_printed_judge_example = False

def run_match(pair, model_name, model_id, with_context, support_thinking, use_thinking, api_key, api_url, max_tokens=8192, max_retries=3):
    global _printed_judge_example
    prompt = build_judge_prompt(pair, with_context=with_context)
    if not _printed_judge_example:
        _printed_judge_example = True
        print("\n" + "="*80)
        print(f"EXAMPLE JUDGE PROMPT (model={model_name}, context={with_context})")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
    raw_answer, usage = None, {}
    judge_answer, bad_round_pred, behavior_type_pred = None, None, None

    for attempt in range(1, max_retries + 1):
        try:
            raw_answer, usage = call_openrouter(
                model_id, prompt, api_key, api_url, max_tokens=max_tokens,
                response_format=JUDGE_RESP_FORMAT,
                support_thinking=support_thinking, use_thinking=use_thinking)
            if not raw_answer:
                print(f"  empty_response attempt {attempt}/{max_retries} for {model_name} on {pair['id']}, retrying...", flush=True)
                continue
            # Strip markdown code fences (```json ... ```) that some models wrap around JSON
            clean = re.sub(r'^```(?:json)?\s*', '', raw_answer.strip())
            clean = re.sub(r'\s*```$', '', clean)
            parsed = json.loads(clean)
            judge_answer = parsed["verdict"]
            worst_round = parsed["worst_round"]
            bad_round_pred = int(worst_round) if worst_round is not None else None
            behavior_type_pred = LETTER_TO_BEHAVIOR.get(parsed["problem_type"], parsed["problem_type"])
            behavior_type_pred = LETTER_TO_BEHAVIOR.get(parsed["problem_type"], parsed["problem_type"])
            break
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  parse_fail attempt {attempt}/{max_retries} for {model_name} on {pair['id']}: {e}", flush=True)
        except Exception as e:
            print(f"  api_error attempt {attempt}/{max_retries} for {model_name} on {pair['id']}: {e}", flush=True)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    if _printed_judge_example and not hasattr(run_match, '_printed_response'):
        run_match._printed_response = True
        print("\n" + "="*80)
        print("EXAMPLE JUDGE RESPONSE")
        print("="*80)
        print(raw_answer)
        print("="*80 + "\n")

    # --- Correctness ---
    better_is_a = pair["better_is_a"]
    correct_verdict = (judge_answer == ("A" if better_is_a else "B")) if judge_answer else None

    meta = pair.get("metadata", {})
    bad_round_gt = meta.get("bad_round_index") or pair.get("bad_round_index")
    behavior_type_gt = meta.get("assistant_behavior_type") or pair.get("assistant_behavior_type", "")

    correct_bad_round = (bad_round_pred == bad_round_gt) if judge_answer is not None and bad_round_gt is not None and bad_round_pred is not None else None
    correct_behavior_type = (behavior_type_pred == behavior_type_gt) if judge_answer is not None and behavior_type_gt and behavior_type_pred else None

    # Overall: verdict AND round AND behavior type must all be correct
    if judge_answer is not None:
        correct = correct_verdict
        if correct_bad_round is not None:
            correct = correct and correct_bad_round
        if correct_behavior_type is not None:
            correct = correct and correct_behavior_type
    else:
        correct = None

    return {
        "id": pair["id"],
        "domain": pair.get("domain", ""),
        "better_is_a": better_is_a,
        "pair": {
            "assistant_behavior_type": behavior_type_gt,
            "user_behavior_type": meta.get("user_behavior_type") or pair.get("user_behavior_type", ""),
        },
        "model": {
            "name": model_name,
            "openrouter_name": model_id,
            "add_context": with_context,
            "support_thinking": support_thinking,
            "use_thinking": use_thinking,
        },
        "judge": {
            "raw": raw_answer,
            "answer": judge_answer,
            "bad_round_gt": bad_round_gt,
            "bad_round_pred": bad_round_pred,
            "behavior_type_gt": behavior_type_gt,
            "behavior_type_pred": behavior_type_pred,
            "correct_verdict": correct_verdict,
            "correct_bad_round": correct_bad_round,
            "correct_behavior_type": correct_behavior_type,
            "correct": correct,
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cost": usage.get("cost"),
        },
    }


def make_error_match(pid, name, pair, error):
    return {
        "id": pid,
        "domain": pair.get("domain", ""),
        "better_is_a": None,
        "pair": {
            "assistant_behavior_type": pair.get("metadata", {}).get("assistant_behavior_type") or pair.get("assistant_behavior_type", ""),
            "user_behavior_type": pair.get("metadata", {}).get("user_behavior_type") or pair.get("user_behavior_type", ""),
        },
        "model": {
            "name": name,
        },
        "judge": {
            "raw": f"ERROR:{error}",
            "answer": None,
            "correct": None,
        },
        "usage": {
            "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "cost": None,
        },
    }


def run_jobs(jobs, results, args):
    print(f"{len(jobs)} jobs to run, {len(results)} already done.")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_match, pair, name, mid, ctx, think, use_think, args.api_key, args.api_url, args.max_tokens): (pair["id"], name, pair)
            for pair, name, mid, ctx, think, use_think in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            pid, name, pair = futures[future]
            try:
                match = future.result()
            except Exception as e:
                match = make_error_match(pid, name, pair, e)

            results.append(match)
            judge_correct = match["judge"]["correct"]
            status = "correct" if judge_correct else ("wrong" if judge_correct is False else "parse_fail")
            label = pid if len(str(pid)) <= 16 else str(pid)[:12] + "..."
            ptype = pair.get("metadata", {}).get("assistant_behavior_type") or pair.get("assistant_behavior_type", "")
            print(f"[{i}/{len(jobs)}] {label} {name:14s} | answer={match['judge']['answer']} {status} ({ptype})", flush=True)

            if i % args.save_interval == 0:
                save_matches(args.matches_out, results)
                print(f"Saved -> {args.matches_out}", flush=True)
    return results



def parse_args():
    p = argparse.ArgumentParser(description="RankJudge match runner")
    p.add_argument("--api-key-path", default="../api_key.json")
    p.add_argument("--api-url", default="https://openrouter.ai/api/v1/chat/completions")
    p.add_argument("--pairs", default="../outputs/mode_b/pairs_filtered.json")
    p.add_argument("--matches-out", default="../outputs/mode_b/matches.json")
    p.add_argument("--metrics-out", default="../outputs/mode_b/metrics.json")
    p.add_argument("--init-elo", type=int, default=1500)
    p.add_argument("--top-pct", type=float, default=0.05,
                   help="Fraction of top-Elo pairs to remove for the top_removed slice")
    p.add_argument("--no-top-removed", action="store_true",
                   help="Skip the top_removed slice")
    p.add_argument("--max-tokens", type=int, default=1024*32)
    p.add_argument("--workers", type=int, default=50)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.api_key = load_api_key(args.api_key_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.matches_out)), exist_ok=True)

    judge_configs = build_judge_configs(MODELS)
    with open(args.pairs) as f:
        pairs = json.load(f)
    results, done_keys = load_state(args)

    jobs = build_jobs(pairs, judge_configs, done_keys)
    results = run_jobs(jobs, results, args)
    save_matches(args.matches_out, results)
    compute_metrics(results, args)


if __name__ == "__main__":
    main()

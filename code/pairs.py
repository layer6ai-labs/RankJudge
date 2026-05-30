"""Generate conversation pairs for RankJudge."""

import argparse, hashlib, json, os, random
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import call_structured, load_api_key, load_ml, load_med, load_fin, seed_everything
from prompts import GOOD_CONVO, WORSE_CONVO
from taxonomy import ASSISTANT_BEHAVIORS, USER_BEHAVIORS
from schemas import GOOD_CONVO_SCHEMA, WORSE_CONVO_SCHEMA
from sampler import sample_params


def generate_good_convo(paper, n_rounds, model_id, api_key, api_url, max_tokens,
                        user_behavior_type, assistant_virtue, flaw_name, assistant_flaw):
    """Generate a high-quality multi-turn conversation grounded in the item's context."""
    prompt = GOOD_CONVO.format(
        context=paper["context"], n=n_rounds,
        user_behavior_name=user_behavior_type,
        user_behavior_desc=USER_BEHAVIORS[user_behavior_type],
        assistant_virtue=assistant_virtue,
        flaw_name=flaw_name,
        assistant_flaw=assistant_flaw)
    result, _ = call_structured(model_id, prompt, api_key, api_url, GOOD_CONVO_SCHEMA, max_tokens)
    return result["conversation"], result.get("plan", "")


def generate_worse_convo(paper, n_rounds, model_id, api_key, api_url, max_tokens,
                         behavior_type, user_behavior_type):
    """Generate a conversation with a specific assistant weakness injected into one turn."""
    prompt = WORSE_CONVO.format(
        context=paper["context"], n=n_rounds,
        behavior_type=behavior_type,
        behavior_desc=ASSISTANT_BEHAVIORS[behavior_type]["flaw"],
        user_behavior_name=user_behavior_type,
        user_behavior_desc=USER_BEHAVIORS[user_behavior_type])
    result, _ = call_structured(model_id, prompt, api_key, api_url, WORSE_CONVO_SCHEMA, max_tokens)
    plan = result.get("plan", {})
    return result["conversation"], plan.get("reasoning", ""), plan.get("bad_round_index")


def validate_convo(convo, n_rounds, label):
    """Check that a conversation has exactly n_rounds rounds (user+assistant pairs)."""
    expected_turns = n_rounds * 2
    actual_turns = len(convo)
    if actual_turns != expected_turns:
        raise ValueError(f"{label}: expected {n_rounds} rounds ({expected_turns} turns), got {actual_turns} turns")



def make_id(domain, raw_id):
    return hashlib.sha256(f"{domain}:{raw_id}".encode()).hexdigest()[:12]


def process_paper(paper, domain, model_id, api_key, api_url, max_tokens, uniform=True, max_retries=3):
    """Create a good/bad conversation pair for one item with randomized behaviors and ordering."""
    params = sample_params(uniform=uniform)

    for attempt in range(1, max_retries + 1):
        try:
            good_convo, good_plan = generate_good_convo(
                paper, params["n_rounds"], model_id, api_key, api_url, max_tokens,
                params["user_behavior_type"],
                ASSISTANT_BEHAVIORS[params["assistant_behavior_type"]]["virtue"],
                params["assistant_behavior_type"],
                ASSISTANT_BEHAVIORS[params["assistant_behavior_type"]]["flaw"])
            validate_convo(good_convo, params["n_rounds"], "good_convo")
            break
        except ValueError as e:
            print(f"  [{make_id(domain, paper['id'])}] good_convo validation failed (attempt {attempt}/{max_retries}): {e}", flush=True)
            if attempt == max_retries:
                raise

    for attempt in range(1, max_retries + 1):
        try:
            bad_convo, bad_plan, bad_round_index = generate_worse_convo(
                paper, params["n_rounds"], model_id, api_key, api_url, max_tokens,
                params["assistant_behavior_type"], params["user_behavior_type"])
            validate_convo(bad_convo, params["n_rounds"], "bad_convo")
            break
        except ValueError as e:
            print(f"  [{make_id(domain, paper['id'])}] bad_convo validation failed (attempt {attempt}/{max_retries}): {e}", flush=True)
            if attempt == max_retries:
                raise

    pid = make_id(domain, paper["id"])
    print(f"  [{pid}] validation passed: {params['n_rounds']} rounds, bad_round_index={bad_round_index}", flush=True)

    better_is_a = random.random() < 0.5
    convo_a, convo_b = (good_convo, bad_convo) if better_is_a else (bad_convo, good_convo)
    return {
        "id": pid,
        "domain": domain,
        "convo_a": convo_a, "convo_b": convo_b,
        "better_is_a": better_is_a,
        "plan": {"good": good_plan, "bad": bad_plan},
        "metadata": {
            "user_behavior_type": params["user_behavior_type"],
            "assistant_behavior_type": params["assistant_behavior_type"],
            "n_rounds": params["n_rounds"],
            "bad_round_index": bad_round_index,
            "context": paper["context"],
        },
    }


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


def run_generation(papers, domain, results, done_ids, args):
    """Process all items in parallel and write conversation pairs to output file."""
    papers = [p for p in papers if make_id(domain, p["id"]) not in done_ids]
    print(f"Processing {len(papers)} {domain} papers with {args.model}...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_paper, p, domain, args.model, args.api_key,
                          args.api_url, args.max_tokens, args.uniform): p for p in papers}
        for i, fut in enumerate(as_completed(futs), 1):
            paper = futs[fut]
            try:
                results.append(fut.result())
                print(f"[{i}/{len(papers)}] {paper['id']} - done ({len(results)} pairs)", flush=True)
            except Exception as e:
                print(f"[{i}/{len(papers)}] {paper['id']} - ERROR: {e}", flush=True)
            if i % args.save_interval == 0:
                save_pairs(args.out, results)
                print(f"Saved -> {args.out}", flush=True)
    save_pairs(args.out, results)
    print(f"Done {domain}. {len(results)} total pairs -> {args.out}")


def parse_args():
    """Parse command-line arguments for pair generation."""
    p = argparse.ArgumentParser(description="Generate conversation pairs for RankJudge")
    p.add_argument("--data-dir", default="../data", help="Directory for cached data")
    p.add_argument("--out", default="../outputs/mode_b/pairs.json")
    p.add_argument("--api-key-path", default="../api_key.json")
    p.add_argument("--api-url", default="https://openrouter.ai/api/v1/chat/completions")
    p.add_argument("--model", default="openai/gpt-5.5")
    p.add_argument("--max-tokens", type=int, default=1024 * 32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--uniform", action=argparse.BooleanOptionalAction, default=True,
                   help="Sample params uniformly over keys (default). Use --no-uniform to follow DISTRIBUTIONS weights in sampler.py.")
    p.add_argument("--dataset", choices=["ml", "med", "fin"], nargs="*", default=["ml", "med", "fin"],
                   help="Domains to run (default: all three). E.g. --dataset ml fin")
    return p.parse_args()


def main():
    """Load papers, seed RNG, and run parallel pair generation."""
    args = parse_args()
    args.api_key = load_api_key(args.api_key_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results, done_ids = load_state(args)
    loaders = {"ml": load_ml, "med": load_med, "fin": load_fin}
    for domain in args.dataset:
        print(f"\n{'='*60}\nDomain: {domain}\n{'='*60}", flush=True)
        seed_everything(args.seed)
        papers = loaders[domain](args.data_dir, args.n_samples)
        run_generation(papers, domain, results, done_ids, args)


if __name__ == "__main__":
    main()

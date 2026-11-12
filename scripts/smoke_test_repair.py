"""Offline smoke test for the RepairJudge pipeline: no API key, no cost, no download.

Replaces every LLM call with a fake that returns well-formed output, then runs the
real pairs -> verify -> matches (dialogue + gold) -> metrics code on 10 synthetic
problems. The judge is random, so the printed accuracies are meaningless; the point
is that every stage runs and the files have the right shape.

    python scripts/smoke_test_repair.py            # writes to outputs/smoke/
"""
import json, os, random, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
OUT = os.path.join(ROOT, "outputs", "smoke")
sys.path.insert(0, CODE)
os.chdir(CODE)

import pairs, verify, matches, metrics  # noqa: E402
from utils import parse_number, format_number  # noqa: E402


def fake_structured(model_id, prompt, api_key, api_url, schema, max_tokens=0, **kw):
    props = schema["properties"]
    if "mistake" in props:                                   # wrong first solution
        w = re.search(r"must end with the answer (\S+)\.", prompt).group(1)
        return {"mistake": "stopped early", "solution": f"Some steps.\nThe answer is {w}.", "final_answer": w}, {}
    if "good_response" in props:                             # final turn pair
        m = re.search(r"where N is (\S+) for the good response and (\S+) for the bad", prompt)
        return {"plan": "p", "good_response": f"Let me re-check each step carefully. The answer is {m.group(1)}.",
                "bad_response": f"Let me recompute that one more time. The answer is {m.group(2)}."}, {}
    if "good_stance" in props:                               # verification audit
        right = "fail to revise" in prompt
        g = re.search(r"GOOD final response ===\n.*The answer is (\S+)\.", prompt).group(1)
        b = re.search(r"BAD final response ===\n.*The answer is (\S+)\.", prompt).group(1)
        return {"good_final_answer": g, "bad_final_answer": b,
                "good_stance": "adopts_user" if right else "keeps_original",
                "bad_stance": "keeps_original" if right else "adopts_user",
                "good_reasoning_valid": True, "bad_other_flaws": False, "issue": ""}, {}
    raise ValueError("unexpected schema")


def fake_openrouter(model_id, prompt, api_key, api_url, max_tokens=0, response_format=None, **kw):
    return json.dumps({"analysis": "fake", "worst_round": 2, "problem_type": random.choice("HIB"),
                       "verdict": random.choice("AB")}), {}


def run(mod, argv):
    sys.argv = [mod.__name__] + argv
    mod.main()


def main():
    random.seed(0)
    os.makedirs(os.path.join(OUT, "input"), exist_ok=True)
    problems = [{"id": f"fake-{i}", "question": f"Sam has {i + 2} apples and buys {i + 5} more. How many now?",
                 "gold_answer": str(2 * i + 7), "gold_solution": "",
                 "gold_rationale": f"{i + 2} + {i + 5} = {2 * i + 7}", "context": "q"} for i in range(10)]
    with open(os.path.join(OUT, "input", "gsm8k.json"), "w") as f:
        json.dump(problems, f)
    key = os.path.join(OUT, "fake_key.json")
    with open(key, "w") as f:
        json.dump({"api_key": "fake"}, f)

    pairs.call_structured = verify.call_structured = fake_structured
    matches.call_openrouter = fake_openrouter

    run(pairs, ["--n-samples", "10", "--data-dir", OUT, "--out", f"{OUT}/pairs.json",
                "--api-key-path", key, "--workers", "4"])
    run(verify, ["--pairs", f"{OUT}/pairs.json", "--out", f"{OUT}/verification.json",
                 "--filtered-out", f"{OUT}/pairs_filtered.json", "--api-key-path", key, "--workers", "4"])
    for cond in ("dialogue", "gold"):
        run(matches, ["--pairs", f"{OUT}/pairs_filtered.json", "--matches-out", f"{OUT}/matches_{cond}.json",
                      "--metrics-out", f"{OUT}/metrics_{cond}.json", "--api-key-path", key,
                      "--exclude-model", "gemma3:12b", "--workers", "4"] + (["--gold"] if cond == "gold" else []))
    run(metrics, ["--repair", "--matches", f"{OUT}/matches_dialogue.json", f"{OUT}/matches_gold.json",
                  "--metrics-out", f"{OUT}/metrics.json"])

    n_pairs = len(json.load(open(f"{OUT}/pairs_filtered.json")))
    assert n_pairs == 20, n_pairs
    m = json.load(open(f"{OUT}/metrics.json"))
    assert m["baselines"]["capitulation"]["agree_with_user"]["winner_acc"] == 0.0
    assert m["baselines"]["failure_to_revise"]["agree_with_user"]["winner_acc"] == 1.0
    assert set(m["conditions"]) == {"dialogue", "gold"}
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()

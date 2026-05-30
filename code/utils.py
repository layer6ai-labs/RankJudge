"""
Shared helpers for the benchmark suite.
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
import requests

def load_api_key(path):
    with open(path) as f:
        return json.load(f)["api_key"]


# --- Reproducibility ---

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --- IO helpers ---

def load_jsonl(path):
    """Load JSON array or pretty-printed JSONL file."""
    with open(path) as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    pairs = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        pos = next((i for i in range(pos, len(text)) if text[i] in '{['), len(text))
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        pairs.append(obj)
        pos = end
    return pairs


def call_openrouter(model_id, prompt, api_key, api_url, max_tokens=32768,
                    response_format=None, timeout=480, support_thinking=False, use_thinking=None):
    """Call OpenRouter API and return parsed response content."""
    payload = {"model": model_id,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens}
    if response_format:
        payload["response_format"] = response_format
    if use_thinking is None:
        use_thinking = support_thinking
    if support_thinking and use_thinking:
        payload["reasoning"] = {"enabled": True}
    elif support_thinking and not use_thinking:
        payload["reasoning"] = {"effort": "none"}
    r = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}"},
                      json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if "choices" not in body:
        error_msg = body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else str(body.get("error", body))
        raise RuntimeError(f"API error from {model_id}: {error_msg}")
    content = body["choices"][0]["message"].get("content", "")
    usage = body.get("usage", {})
    return content, usage


def call_structured(model_id, prompt, api_key, api_url, schema, max_tokens=32768, max_retries=3,
                    support_thinking=False, use_thinking=None):
    """Call OpenRouter with structured JSON output and retry on parse failures."""
    resp_format = {"type": "json_schema",
                   "json_schema": {"name": "response", "strict": True, "schema": schema}}
    for attempt in range(1, max_retries + 1):
        try:
            content, usage = call_openrouter(model_id, prompt, api_key, api_url,
                                             max_tokens=max_tokens, response_format=resp_format,
                                             support_thinking=support_thinking, use_thinking=use_thinking)
            return json.loads(content), usage
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"  parse_fail attempt {attempt}/{max_retries}: {e}", flush=True)
            if attempt == max_retries:
                raise


# --- Rating algorithms ---

def collect_matches(results, correctness_key="correct"):
    return [(r["model"]["name"], r["id"], r["judge"][correctness_key])
            for r in results if r.get("judge", {}).get(correctness_key) is not None]


def run_bradley_terry(matches, init_elo, max_iter=1000, tol=1e-6):
    # `correct` may be bool or float in [0, 1]; fractional values support tie-as-half-credit.
    players = set()
    for judge, pid, _ in matches:
        players.add(f"judge:{judge}")
        players.add(f"pair:{pid}")
    players = sorted(players)
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = np.zeros(n)
    games = np.zeros((n, n))
    for judge, pid, correct in matches:
        ji, pi = idx[f"judge:{judge}"], idx[f"pair:{pid}"]
        c = float(correct)
        wins[ji] += c
        wins[pi] += 1.0 - c
        games[ji][pi] += 1
        games[pi][ji] += 1
    p = np.ones(n)
    for _ in range(max_iter):
        p_old = p.copy()
        for i in range(n):
            denom = sum(games[i][j] / (p[i] + p[j]) for j in range(n) if games[i][j] > 0)
            p[i] = wins[i] / denom if denom > 0 else p[i]
        p /= p.mean()
        if np.max(np.abs(p - p_old)) < tol:
            break
    p = np.maximum(p, 1e-10)

    # --- Cluster-robust sandwich SE on β = log p, clustered by pair ---
    beta = np.log(p)
    I_mat = np.zeros((n, n))
    cluster_scores = {}
    for judge, pid, correct in matches:
        ji, pj = idx[f"judge:{judge}"], idx[f"pair:{pid}"]
        eta = beta[ji] - beta[pj]
        sig = 1.0 / (1.0 + np.exp(-eta))
        w = sig * (1.0 - sig)
        I_mat[ji, ji] += w; I_mat[pj, pj] += w
        I_mat[ji, pj] -= w; I_mat[pj, ji] -= w
        r = float(correct) - sig
        s = cluster_scores.setdefault(pid, np.zeros(n))
        s[ji] += r; s[pj] -= r
    B_mat = sum(np.outer(s, s) for s in cluster_scores.values())
    rank = np.linalg.matrix_rank(I_mat, tol=1e-8)
    if rank < n - 1:
        print(f"  [warn] BT info matrix rank {rank} < n-1={n-1}; disconnected components - SE across components is undefined")
    I_pinv = np.linalg.pinv(I_mat, hermitian=True, rcond=1e-10)
    V = I_pinv @ B_mat @ I_pinv
    se_elo = (400.0 / np.log(10.0)) * np.sqrt(np.clip(np.diag(V), 0.0, None))

    elo = {players[i]: round(400 * np.log10(p[i]) + init_elo) for i in range(n)}
    se = {players[i]: float(se_elo[i]) for i in range(n)}
    return elo, se


def compute_win_rates(matches):
    """Compute win rates for judges and pairs from match tuples."""
    judge_wins, judge_total = defaultdict(int), defaultdict(int)
    pair_wins, pair_total = defaultdict(int), defaultdict(int)
    for judge, pid, correct in matches:
        jk, pk = f"judge:{judge}", f"pair:{pid}"
        judge_total[jk] += 1
        pair_total[pk] += 1
        if correct:
            judge_wins[jk] += 1
        else:
            pair_wins[pk] += 1
    win_rates = {}
    for k in judge_total:
        win_rates[k] = round(judge_wins[k] / judge_total[k], 4)
    for k in pair_total:
        win_rates[k] = round(pair_wins[k] / pair_total[k], 4)
    return win_rates


def split_and_sort(elo, win_rates, se=None):
    judges, pairs = {}, {}
    for k, v in elo.items():
        d = {"elo": v, "win_rate": win_rates.get(k, 0)}
        if se is not None and k in se:
            d["elo_se"] = round(se[k], 2)
            d["elo_ci95"] = round(1.96 * se[k], 2)
        (judges if k.startswith("judge:") else pairs)[k] = d
    return {
        "judges": dict(sorted(judges.items(), key=lambda x: -x[1]["elo"])),
        "pairs": dict(sorted(pairs.items(), key=lambda x: -x[1]["elo"])),
    }


# --- Data loading ---

def load_ml(data_dir, n_samples=None):
    """Load preprocessed ML (RPC-Bench) data. Run preprocessing/ml.py first."""
    path = os.path.join(data_dir, "input", "ml.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: cd code && python preprocessing/ml.py")
    print(f"Loading {path}", flush=True)
    papers = load_jsonl(path)
    if n_samples:
        papers = papers[:n_samples]
    return papers


def load_med(data_dir, n_samples=None):
    """Load preprocessed Med (PubMedQA) data. Run preprocessing/med.py first."""
    path = os.path.join(data_dir, "input", "med.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: cd code && python preprocessing/med.py")
    print(f"Loading {path}", flush=True)
    papers = load_jsonl(path)
    if n_samples:
        papers = papers[:n_samples]
    return papers


def load_fin(data_dir, n_samples=None):
    """Load preprocessed Fin (S&P 500 10-K) data. Run preprocessing/fin.py first."""
    path = os.path.join(data_dir, "input", "fin.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: cd code && python preprocessing/fin.py")
    print(f"Loading {path}", flush=True)
    papers = load_jsonl(path)
    if n_samples:
        papers = papers[:n_samples]
    return papers

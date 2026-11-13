"""
Shared helpers for the benchmark suite.
"""

import json
import os
import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation

import numpy as np
import requests

# --- Backends ---
# Every model is called through an OpenAI-compatible /v1/chat/completions endpoint on a
# local model server (vLLM on PACE: LOCAL_API_URL=http://localhost:8000/v1/chat/completions).
LOCAL_API_URL = os.environ.get("LOCAL_API_URL", "http://localhost:11434/v1/chat/completions")
# OpenRouter API (from RankJudge) -- NOT USED: we only run local models on PACE.
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def resolve_api_url(url):
    """'local' / a full URL -> a full URL."""
    if url in (None, "", "local"):
        return LOCAL_API_URL
    # OpenRouter API (from RankJudge) -- NOT USED: we only run local models on PACE.
    # if url == "openrouter":
    #     return OPENROUTER_API_URL
    return url


def is_openrouter(api_url):
    return "openrouter.ai" in (api_url or "")


def load_api_key(path):
    with open(path) as f:
        return json.load(f)["api_key"]


def get_api_key(path, api_urls):
    """API key only if some call goes to OpenRouter; local servers need none."""
    if not any(is_openrouter(u) for u in api_urls):
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. OpenRouter needs an API key (not used in this project).")
    return load_api_key(path)


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
    """Call an OpenAI-compatible chat endpoint (our local model server; the name is from RankJudge)."""
    payload = {"model": model_id,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens}
    if response_format:
        payload["response_format"] = response_format
    if use_thinking is None:
        use_thinking = support_thinking
    # OpenRouter API (from RankJudge) -- NOT USED: we only run local models on PACE.
    # if is_openrouter(api_url):   # "reasoning" is an OpenRouter-only field
    #     if support_thinking and use_thinking:
    #         payload["reasoning"] = {"enabled": True}
    #     elif support_thinking and not use_thinking:
    #         payload["reasoning"] = {"effort": "none"}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
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
    """Call a chat endpoint for a JSON object matching `schema`; retry on parse failures.

    OpenRouter: the schema is enforced by the server (response_format=json_schema).
    Local servers: schema-constrained decoding makes gpt-oss on vLLM emit endless
    whitespace, so by default the schema goes into the prompt instead and the reply is
    coerced to it (coerce_to_schema). Set LOCAL_JSON_SCHEMA=1 to enforce it server-side.
    """
    if is_openrouter(api_url) or os.environ.get("LOCAL_JSON_SCHEMA") == "1":
        resp_format = {"type": "json_schema",
                       "json_schema": {"name": "response", "strict": True, "schema": schema}}
    else:
        resp_format = None
        prompt = (f"{prompt}\n\nReturn ONLY one JSON object (no code fences, no other text) "
                  f"that matches this JSON schema. Use exactly the listed enum values.\n"
                  f"{json.dumps(schema)}")
    for attempt in range(1, max_retries + 1):
        try:
            content, usage = call_openrouter(model_id, prompt, api_key, api_url,
                                             max_tokens=max_tokens, response_format=resp_format,
                                             support_thinking=support_thinking, use_thinking=use_thinking)
            result = parse_json_reply(content)
            return (result if resp_format else coerce_to_schema(result, schema)), usage
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
            print(f"  parse_fail attempt {attempt}/{max_retries}: {e}", flush=True)
            if attempt == max_retries:
                raise


def coerce_to_schema(obj, schema):
    """Fix the usual near-misses of an unconstrained reply against a flat object schema.

    52 -> "52" for strings; "keeps" -> "keeps_original" for enums (unique prefix match);
    null / "true" / "no" -> bool for booleans. Raises KeyError for a missing required
    field and ValueError for an enum value that cannot be matched, so the call retries.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    out = {}
    for key, spec in schema.get("properties", {}).items():
        if key not in obj:
            if key in schema.get("required", []):
                raise KeyError(f"missing field {key!r}")
            continue
        v = obj[key]
        if "enum" in spec:
            s = str(v).strip().lower()
            hits = [e for e in spec["enum"] if e == s] or [e for e in spec["enum"] if e.startswith(s) and s]
            if len(hits) != 1:
                raise ValueError(f"{key}={v!r} not in {spec['enum']}")
            v = hits[0]
        elif spec.get("type") == "boolean":
            if not isinstance(v, bool):
                v = str(v).strip().lower() in ("true", "yes", "1")   # null / "" / "no" -> False
        elif spec.get("type") == "string":
            v = "" if v is None else (v if isinstance(v, str) else json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        out[key] = v
    return out


def parse_json_reply(content):
    """json.loads that tolerates ```json fences and text around the object (small local models)."""
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # strict=False: accept raw newlines/tabs inside strings, which local models often emit
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1], strict=False)
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

def load_gsm8k(data_dir, n_samples=None):
    """Load preprocessed GSM8K problems. Run preprocessing/gsm8k.py first."""
    path = os.path.join(data_dir, "input", "gsm8k.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: cd code && python preprocessing/gsm8k.py")
    print(f"Loading {path}", flush=True)
    problems = load_jsonl(path)
    if n_samples:
        problems = problems[:n_samples]
    return problems


# --- Numeric helpers (GSM8K) ---


_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?|-?\$?\.\d+")


def parse_number(s):
    """Parse '1,234', '$5.50', '18', '-3' -> Decimal. Returns None if unparseable."""
    if s is None:
        return None
    if isinstance(s, (int, float, Decimal)):
        return Decimal(str(s))
    t = str(s).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return Decimal(t)
    except InvalidOperation:
        pass
    # Models sometimes wrap the value in stray punctuation (e.g. Gemma's "，10" or "。8").
    # Accept it only when exactly one number is present.
    m = re.fullmatch(r"\D*?(-?\d+(?:\.\d+)?)\D*", t)
    return Decimal(m.group(1)) if m else None


def format_number(d):
    """Render a Decimal without trailing zeros ('18.00' -> '18')."""
    d = parse_number(d)
    if d is None:
        return None
    if d == d.to_integral_value():
        return str(int(d))
    return format(d.normalize(), "f")


def numbers_equal(a, b, tol=Decimal("1e-6")):
    a, b = parse_number(a), parse_number(b)
    return a is not None and b is not None and abs(a - b) <= tol


def extract_final_number(text):
    """Best-effort final answer from free text.

    Prefers an explicit 'answer is X' / '#### X' / bold **X** statement; falls back to
    the last number in the text. Returns a Decimal or None.
    """
    if not text:
        return None
    for pat in (r"####\s*(" + _NUM_RE.pattern + ")",
                r"answer\s*(?:is|:|=|remains|should be|would be)\s*\**\s*(" + _NUM_RE.pattern + ")",
                r"\*\*\s*(" + _NUM_RE.pattern + r")\s*\**"):
        hits = re.findall(pat, text, flags=re.I)
        if hits:
            return parse_number(hits[-1])
    nums = _NUM_RE.findall(text)
    return parse_number(nums[-1]) if nums else None

#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint -- pick the path with the first argument:
#
#   ./run.sh a   Mode A: materialize the released Layer6/RankJudge dataset into
#                outputs/mode_a/ (pairs.json, matches.json) and compute
#                metrics.json directly from it. Extra args are forwarded to
#                rank_from_hf.py, e.g.  ./run.sh a --split train
#
#   ./run.sh b   Mode B (default): regenerate from scratch -- pairs -> verify ->
#                matches -> metrics, into outputs/mode_b/. Configured below.

MODE="${1:-b}"
if [[ $# -gt 0 ]]; then shift; fi

# ==== run from code/ ====
cd "$(dirname "$0")/../code"

case "$MODE" in
  a|A)
    exec python rank_from_hf.py "$@"
    ;;
  b|B)
    : # fall through to the Mode B pipeline below
    ;;
  *)
    echo "usage: $(basename "$0") [a|b] [extra flags forwarded to Mode A]" >&2
    exit 2
    ;;
esac

# ======================== Mode B: regenerate from scratch ========================

# ==== config ====
DATASETS=(ml med fin)
N_SAMPLES=100
SEED=42

PAIRS_MODEL="openai/gpt-5.5"
PAIRS_WORKERS=50
PAIRS_MAX_TOKENS=32768

VERIFY_MODEL="openai/gpt-5.5"
VERIFY_WORKERS=50
VERIFY_MAX_TOKENS=32768

MATCHES_WORKERS=100
MATCHES_MAX_TOKENS=32768

INIT_ELO=1500
TOP_PCT=0.05

API_KEY_PATH="../api_key.json"
API_URL="https://openrouter.ai/api/v1/chat/completions"

PAIRS_OUT="../outputs/mode_b/pairs.json"
VERIFY_OUT="../outputs/mode_b/verification.json"
PAIRS_FILTERED="../outputs/mode_b/pairs_filtered.json"
MATCHES_OUT="../outputs/mode_b/matches.json"
METRICS_OUT="../outputs/mode_b/metrics.json"

# stages: 1=run, 0=skip
RUN_PREPROCESS=1   # 1 on first run (downloads raw + builds data/input); set 0 once cached
RUN_PAIRS=1
RUN_VERIFY=1
RUN_MATCHES=1
RUN_METRICS=0
RESUME=1

if [[ $RUN_PREPROCESS -eq 1 ]]; then
  for ds in "${DATASETS[@]}"; do
    python "preprocessing/${ds}.py"
  done
fi

RESUME_FLAG=$([[ $RESUME -eq 1 ]] && echo "--resume" || echo "--no-resume")

if [[ $RUN_PAIRS -eq 1 ]]; then
  python pairs.py \
    --dataset "${DATASETS[@]}" \
    --n-samples "$N_SAMPLES" \
    --workers "$PAIRS_WORKERS" \
    --model "$PAIRS_MODEL" \
    --max-tokens "$PAIRS_MAX_TOKENS" \
    --seed "$SEED" \
    --out "$PAIRS_OUT" \
    --api-key-path "$API_KEY_PATH" \
    --api-url "$API_URL" \
    $RESUME_FLAG
fi

if [[ $RUN_VERIFY -eq 1 ]]; then
  python verify.py \
    --pairs "$PAIRS_OUT" \
    --out "$VERIFY_OUT" \
    --filtered-out "$PAIRS_FILTERED" \
    --model "$VERIFY_MODEL" \
    --workers "$VERIFY_WORKERS" \
    --max-tokens "$VERIFY_MAX_TOKENS" \
    --api-key-path "$API_KEY_PATH" \
    --api-url "$API_URL" \
    $RESUME_FLAG
fi

if [[ $RUN_MATCHES -eq 1 ]]; then
  python matches.py \
    --pairs "$PAIRS_FILTERED" \
    --matches-out "$MATCHES_OUT" \
    --metrics-out "$METRICS_OUT" \
    --workers "$MATCHES_WORKERS" \
    --max-tokens "$MATCHES_MAX_TOKENS" \
    --init-elo "$INIT_ELO" \
    --top-pct "$TOP_PCT" \
    --seed "$SEED" \
    --api-key-path "$API_KEY_PATH" \
    --api-url "$API_URL" \
    $RESUME_FLAG
fi

if [[ $RUN_METRICS -eq 1 ]]; then
  python metrics.py \
    --matches "$MATCHES_OUT" \
    --metrics-out "$METRICS_OUT" \
    --init-elo "$INIT_ELO" \
    --top-pct "$TOP_PCT"
fi

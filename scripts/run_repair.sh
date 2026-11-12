#!/usr/bin/env bash
set -euo pipefail

# RepairJudge pipeline (GSM8K, user-correction pairs). Run from the repo root:
#   bash scripts/run_repair.sh                                       # full run, local models
#   N_PROBLEMS=10 OUT=../outputs/pilot bash scripts/run_repair.sh    # small trial run
#   BACKEND=openrouter bash scripts/run_repair.sh                    # use the OpenRouter API instead
# Every stage resumes from its output file, so re-running after a crash only does
# the missing work. Toggle stages with the RUN_* flags below.

cd "$(dirname "$0")/../code"

# ==== config ====
N_PROBLEMS="${N_PROBLEMS:-300}"     # GSM8K problems -> up to 2 pairs each
OUT="${OUT:-../outputs/repair}"     # relative to code/
BACKEND="${BACKEND:-local}"         # local (Ollama/vLLM) | openrouter
SEED=42
MAX_TOKENS=4096

if [[ "$BACKEND" == "local" ]]; then
  GEN_MODEL="${GEN_MODEL:-gemma3:12b}"        # generator; automatically excluded from the judges
  VERIFY_MODEL="${VERIFY_MODEL:-$GEN_MODEL}"
  WORKERS="${WORKERS:-2}"                     # a local server handles a few requests at a time
else
  GEN_MODEL="${GEN_MODEL:-openai/gpt-5.5}"
  VERIFY_MODEL="${VERIFY_MODEL:-$GEN_MODEL}"
  WORKERS="${WORKERS:-20}"
fi
# Judges: empty = the five in JUDGE_SETS[$BACKEND] (code/models.py), or list judge_name values.
JUDGES=()

API_KEY_PATH="../api_key.json"      # only read when BACKEND=openrouter

# stages: 1=run, 0=skip
RUN_PREPROCESS=1
RUN_PAIRS=1
RUN_VERIFY=1
RUN_JUDGE_DIALOGUE=1
RUN_JUDGE_GOLD=1
RUN_METRICS=1

mkdir -p "$OUT"
JUDGE_FLAG=()
if [[ ${#JUDGES[@]} -gt 0 ]]; then JUDGE_FLAG=(--judges "${JUDGES[@]}"); fi
COMMON=(--backend "$BACKEND" --workers "$WORKERS" --max-tokens "$MAX_TOKENS" --api-key-path "$API_KEY_PATH" --resume)

echo "Backend: $BACKEND | generator: $GEN_MODEL | problems: $N_PROBLEMS | out: $OUT"

if [[ $RUN_PREPROCESS -eq 1 ]]; then
  python preprocessing/gsm8k.py --n "$N_PROBLEMS" --seed "$SEED"
fi

if [[ $RUN_PAIRS -eq 1 ]]; then
  python pairs.py --n-samples "$N_PROBLEMS" --seed "$SEED" --model "$GEN_MODEL" \
    --out "$OUT/pairs.json" "${COMMON[@]}"
fi

if [[ $RUN_VERIFY -eq 1 ]]; then
  python verify.py --pairs "$OUT/pairs.json" --out "$OUT/verification.json" \
    --filtered-out "$OUT/pairs_filtered.json" --model "$VERIFY_MODEL" "${COMMON[@]}"
fi

if [[ $RUN_JUDGE_DIALOGUE -eq 1 ]]; then
  python matches.py --pairs "$OUT/pairs_filtered.json" \
    --matches-out "$OUT/matches_dialogue.json" --metrics-out "$OUT/metrics_dialogue.json" \
    --exclude-model "$GEN_MODEL" "${JUDGE_FLAG[@]+"${JUDGE_FLAG[@]}"}" "${COMMON[@]}"
fi

if [[ $RUN_JUDGE_GOLD -eq 1 ]]; then
  python matches.py --pairs "$OUT/pairs_filtered.json" --gold \
    --matches-out "$OUT/matches_gold.json" --metrics-out "$OUT/metrics_gold.json" \
    --exclude-model "$GEN_MODEL" "${JUDGE_FLAG[@]+"${JUDGE_FLAG[@]}"}" "${COMMON[@]}"
fi

if [[ $RUN_METRICS -eq 1 ]]; then
  python metrics.py --repair \
    --matches "$OUT/matches_dialogue.json" "$OUT/matches_gold.json" \
    --metrics-out "$OUT/metrics.json"
fi

#!/usr/bin/env bash
set -euo pipefail

# RepairJudge pipeline (GSM8K, user-correction pairs). Run from the repo root:
#   bash scripts/run_repair.sh                                       # everything on the model server (PACE, README step 5)
#   N_PROBLEMS=10 OUT=../outputs/pilot bash scripts/run_repair.sh    # small trial run
#   GEN_BACKEND=openrouter bash scripts/run_repair.sh                # generate on OpenRouter (Gemma 4 31B), judge locally
#   BACKEND=openrouter bash scripts/run_repair.sh                    # everything on OpenRouter
# GEN_BACKEND drives pairs + verify, JUDGE_BACKEND drives the judges; both default to BACKEND.
# Every stage resumes from its output file, so re-running after a crash only does
# the missing work. Toggle stages with the RUN_* flags below.

cd "$(dirname "$0")/../code"

# ==== config ====
N_PROBLEMS="${N_PROBLEMS:-300}"     # GSM8K problems -> up to 2 pairs each
OUT="${OUT:-../outputs/repair}"     # relative to code/
BACKEND="${BACKEND:-local}"         # local (model server on this node, e.g. Ollama on PACE) | openrouter
GEN_BACKEND="${GEN_BACKEND:-$BACKEND}"       # pairs + verify
JUDGE_BACKEND="${JUDGE_BACKEND:-$BACKEND}"   # matches (judges)
SEED=42
MAX_TOKENS=4096

if [[ "$GEN_BACKEND" == "local" ]]; then
  GEN_MODEL="${GEN_MODEL:-gpt-oss:120b}"      # generator; automatically excluded from the judges
  GEN_WORKERS="${GEN_WORKERS:-4}"             # matches OLLAMA_NUM_PARALLEL=4 on PACE
else
  GEN_MODEL="${GEN_MODEL:-google/gemma-4-31b-it}"    # cheap; about $1 for 300 problems
  GEN_WORKERS="${GEN_WORKERS:-20}"
fi
VERIFY_MODEL="${VERIFY_MODEL:-$GEN_MODEL}"
if [[ "$JUDGE_BACKEND" == "local" ]]; then
  JUDGE_WORKERS="${JUDGE_WORKERS:-4}"
else
  JUDGE_WORKERS="${JUDGE_WORKERS:-20}"
fi
# Judges: empty = the five in JUDGE_SETS[$JUDGE_BACKEND] (code/models.py), or list judge_name values.
JUDGES=()

API_KEY_PATH="../api_key.json"      # only read for the openrouter backend

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
SHARED=(--max-tokens "$MAX_TOKENS" --api-key-path "$API_KEY_PATH" --resume)
GEN=(--backend "$GEN_BACKEND" --workers "$GEN_WORKERS" "${SHARED[@]}")
JUDGE=(--backend "$JUDGE_BACKEND" --workers "$JUDGE_WORKERS" "${SHARED[@]}")

echo "Generator: $GEN_MODEL ($GEN_BACKEND) | judges: $JUDGE_BACKEND | problems: $N_PROBLEMS | out: $OUT"

if [[ $RUN_PREPROCESS -eq 1 ]]; then
  python preprocessing/gsm8k.py --n "$N_PROBLEMS" --seed "$SEED"
fi

if [[ $RUN_PAIRS -eq 1 ]]; then
  python pairs.py --n-samples "$N_PROBLEMS" --seed "$SEED" --model "$GEN_MODEL" \
    --out "$OUT/pairs.json" "${GEN[@]}"
fi

if [[ $RUN_VERIFY -eq 1 ]]; then
  python verify.py --pairs "$OUT/pairs.json" --out "$OUT/verification.json" \
    --filtered-out "$OUT/pairs_filtered.json" --model "$VERIFY_MODEL" "${GEN[@]}"
fi

if [[ $RUN_JUDGE_DIALOGUE -eq 1 ]]; then
  python matches.py --pairs "$OUT/pairs_filtered.json" \
    --matches-out "$OUT/matches_dialogue.json" --metrics-out "$OUT/metrics_dialogue.json" \
    --exclude-model "$GEN_MODEL" "${JUDGE_FLAG[@]+"${JUDGE_FLAG[@]}"}" "${JUDGE[@]}"
fi

if [[ $RUN_JUDGE_GOLD -eq 1 ]]; then
  python matches.py --pairs "$OUT/pairs_filtered.json" --gold \
    --matches-out "$OUT/matches_gold.json" --metrics-out "$OUT/metrics_gold.json" \
    --exclude-model "$GEN_MODEL" "${JUDGE_FLAG[@]+"${JUDGE_FLAG[@]}"}" "${JUDGE[@]}"
fi

if [[ $RUN_METRICS -eq 1 ]]; then
  python metrics.py --repair \
    --matches "$OUT/matches_dialogue.json" "$OUT/matches_gold.json" \
    --metrics-out "$OUT/metrics.json"
fi

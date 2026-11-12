#!/usr/bin/env bash
set -euo pipefail

# Reproduce the original RankJudge baseline (Mode A): download the released
# Layer6/RankJudge dataset into outputs/mode_a/ (pairs.json, matches.json) and
# compute metrics.json from it. No API calls. Extra args go to rank_from_hf.py:
#   bash scripts/run.sh a
#   bash scripts/run.sh a --split train
#
# Our own pipeline is scripts/run_repair.sh.

if [[ "${1:-a}" == "a" || "${1:-}" == "A" ]]; then
  [[ $# -gt 0 ]] && shift
fi

cd "$(dirname "$0")/../code"
exec python rank_from_hf.py "$@"

#!/usr/bin/env bash
# Residual-SAC generalization evaluation in Dynamic LIBERO v2 (no training).
# Reads a jobs file with one "<run_name> <eval_residual.py args...>" per line and runs
# scripts/eval_residual.py for each, JOBS at a time, round-robin over GPUS.
# Existing reports are skipped (never overwritten).
# Usage: OUT=<dir> GPUS="0 1 2 3" JOBS=8 EPISODES=50 scripts/run_v2_generalization.sh <jobs_file>
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS_FILE="${1:?jobs file}"
export OUT="${OUT:?set OUT}" EPISODES="${EPISODES:-50}"
GPUS=(${GPUS:-0 1 2 3})
mkdir -p "$OUT"
tmp=$(mktemp)
i=0
while read -r name args; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  if [ -e "$OUT/$name.json" ]; then echo "skip existing $name"; continue; fi
  echo "${GPUS[$((i % ${#GPUS[@]}))]} $name $args" >> "$tmp"
  i=$((i + 1))
done < "$JOBS_FILE"
run_one() {
  gpu=$1; name=$2; shift 2
  (source scripts/server_env.sh "$gpu" >/dev/null
   "$PY" -u -B scripts/eval_residual.py "$@" --episodes "$EPISODES" --seed 0 --save-steps \
     --out "$OUT/$name.json" > "$OUT/$name.log" 2>&1) && echo "done $name $(date +%T)" || echo "FAILED $name"
}
export -f run_one
xargs -P "${JOBS:-8}" -L 1 bash -c 'run_one "$@"' _ < "$tmp"
rm -f "$tmp"
echo "ALLDONE $(date)"

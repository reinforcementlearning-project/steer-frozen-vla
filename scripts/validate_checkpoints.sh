#!/usr/bin/env bash
# Pre-registered checkpoint validation (see <run>/validation_rule.json):
# ckpt 2k..20k x (+x,-x,+y,-y at 2cm) x 10 episodes, eval seed 1000, deterministic actor.
# Then selects a checkpoint with scripts/select_validation_checkpoint.py.
# Usage: scripts/validate_checkpoints.sh <run_dir> [parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?run_dir}"
JOBS="${2:-12}"
OUT="$RUN/validation"
mkdir -p "$OUT"
jobs_file=$(mktemp)
i=0
STEPS="${STEPS:-$(seq 2000 1000 20000)}"  # override only for smoke tests
for k in $STEPS; do
  ck=$(printf "%s/checkpoints/ckpt_%08d.pt" "$RUN" "$k")
  [ -e "$ck" ] || { echo "missing $ck" >&2; exit 1; }
  for cond in "px 0 1" "mx 0 -1" "py 1 1" "my 1 -1"; do
    set -- $cond
    f="$OUT/val_$(printf %05d "$k")_${1}2.json"
    [ -e "$f" ] && continue
    echo "$((i % 4)) $f $2 $3 $ck" >> "$jobs_file"
    i=$((i + 1))
  done
done
run_one() {
  gpu=$1; f=$2; axis=$3; dir=$4; ck=$5
  (source scripts/server_env.sh "$gpu" >/dev/null
   "$PY" -u -B scripts/eval_residual.py --checkpoint "$ck" --episodes 10 --seed 1000 \
     --target alphabet_soup_1 --distance-cm 2 --axis "$axis" --direction "$dir" \
     --out "$f" > "${f%.json}.log" 2>&1) && echo "done $f" || echo "FAILED $f"
}
export -f run_one
xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
rm -f "$jobs_file"
source scripts/server_env.sh 0 >/dev/null
CUDA_VISIBLE_DEVICES= "$PY" -B scripts/select_validation_checkpoint.py "$RUN"

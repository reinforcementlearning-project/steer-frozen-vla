#!/usr/bin/env bash
# Direction-generalization evaluation of the seed-42 residual run (experiment A/B).
# 3 policies x 9 perturbation conditions x 50 episodes, eval seed 0 (episode i uses
# reset/noise seed i, identical across policies). Skips reports that already exist.
# Usage: scripts/run_direction_eval.sh [parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=results/residual_2cm_s42_20261124_20k
OUT="${OUT:-results/residual_s42_diagnostics/evals}"  # override for other evaluation sets
# MAX_STEPS (env, optional): episode horizon override for diagnostic runs; default 280 from eval_residual.py
JOBS="${1:-12}"
mkdir -p "$OUT"

# name axis direction distance_cm
CONDITIONS="0cm 1 1 0
py2 1 1 2
my2 1 -1 2
px2 0 1 2
mx2 0 -1 2
py4 1 1 4
my4 1 -1 4
px4 0 1 4
mx4 0 -1 4"
# name policy-args (override with the POLICIES env var, one "name args..." per line)
POLICIES="${POLICIES:-base --zero-residual
ck7k --checkpoint $RUN/checkpoints/ckpt_00007000.pt
final --checkpoint $RUN/checkpoints/ckpt_00020000.pt}"

jobs_file=$(mktemp)
i=0
while read -r pname pargs; do
  while read -r cname axis dir dist; do
    if [ -n "${ONLY:-}" ] && ! grep -qw -- "$cname" <<< "$ONLY"; then continue; fi  # optional condition filter
    f="$OUT/${pname}_${cname}.json"
    [ -e "$f" ] && { echo "skip existing $f"; continue; }
    echo "$((i % 4)) $f $axis $dir $dist $pargs" >> "$jobs_file"
    i=$((i + 1))
  done <<< "$CONDITIONS"
done <<< "$POLICIES"

run_one() {
  gpu=$1; f=$2; axis=$3; dir=$4; dist=$5; shift 5
  (source scripts/server_env.sh "$gpu" >/dev/null
   "$PY" -u -B scripts/eval_residual.py "$@" --episodes 50 --seed 0 ${MAX_STEPS:+--max-steps "$MAX_STEPS"} \
     --target alphabet_soup_1 --distance-cm "$dist" --axis "$axis" --direction "$dir" \
     --save-steps --out "$f" > "${f%.json}.log" 2>&1) && echo "done $f" || echo "FAILED $f"
}
export -f run_one
xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
rm -f "$jobs_file"

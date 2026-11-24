#!/usr/bin/env bash
# Launch one follow-up Residual-SAC variant. Everything except the named change
# matches the seed-42 baseline (results/residual_2cm_s42_20261124_20k).
# Usage: scripts/launch_residual_variant.sh <variant> [gpu] [seed]
#   6d        : 6D arm-only residual (SAC target entropy -6), fixed +y 2cm
#   6d_rand   : 6D residual + per-episode random direction (+x/-x/+y/-y), 2cm
#   6d_rand_fixa001 : 6d_rand + fixed SAC entropy coefficient 0.01 (no auto-tuning)
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:?variant: 6d | 6d_rand}"
gpu="${2:-0}"
seed="${3:-42}"
source scripts/server_env.sh "$gpu"
case "$variant" in
  6d)      extra=(--residual-dims 6) ;;
  6d_rand) extra=(--residual-dims 6 --random-direction) ;;
  6d_rand_fixa001) extra=(--residual-dims 6 --random-direction --fixed-entropy-coef 0.01) ;;
  *) echo "unknown variant $variant" >&2; exit 1 ;;
esac
run_dir="results/residual_${variant}_2cm_s${seed}_20k"
if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite an existing run: $run_dir" >&2
  exit 1
fi
mkdir "$run_dir"
# Pre-registered checkpoint-selection rule, stored with the run before training starts.
cp configs/validation_rule.json "$run_dir/validation_rule.json"
nohup "$PY" -u -B train_residual.py \
  --suite libero_object --task-id 0 --n-action-steps 10 \
  --target alphabet_soup_1 --distance-cm 2 --seed "$seed" \
  --alpha 0.1 --total-env-steps 20000 --warmup-steps 1000 \
  --batch-size 256 --utd-ratio 1 --log-every 100 --checkpoint-every 1000 \
  "${extra[@]}" --run-dir "$run_dir" > "$run_dir/console.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/process.pid"
printf 'PID=%s RUN=%s\n' "$(cat "$run_dir/process.pid")" "$run_dir"

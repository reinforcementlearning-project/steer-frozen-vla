#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh "${1:-0}"
run_dir=results/residual_2cm_s42_20261124_20k
if [ -e "$run_dir" ]; then
  echo "Refusing to overwrite an existing run: $run_dir" >&2
  exit 1
fi
mkdir "$run_dir"
nohup "$PY" -u -B train_residual.py \
  --suite libero_object --task-id 0 --n-action-steps 10 \
  --target alphabet_soup_1 --distance-cm 2 --seed 42 \
  --alpha 0.1 --total-env-steps 20000 --warmup-steps 1000 \
  --batch-size 256 --utd-ratio 1 --log-every 100 --checkpoint-every 1000 \
  --run-dir "$run_dir" > "$run_dir/console.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/process.pid"
printf 'PID=%s RUN=%s\n' "$(cat "$run_dir/process.pid")" "$run_dir"

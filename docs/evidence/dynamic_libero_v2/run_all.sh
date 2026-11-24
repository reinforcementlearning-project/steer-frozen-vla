#!/usr/bin/env bash
# Frozen-base validation rollouts for Dynamic LIBERO v2. GPU 1-3 were occupied by another
# user, so all jobs share one GPU (GPU env var, default 0) with JOBS concurrent processes.
cd "$(dirname "$0")/../../.."
O=results/dynamic_libero_v2/validation
run_one() {
  name=$1; shift
  [ -e "$O/$name.json" ] && { echo "skip $name"; return; }
  ( source scripts/server_env.sh "${GPU:-0}" >/dev/null
    "$PY" -u -B scripts/run_dynamic_libero_v2.py "$@" --out "$O/$name.json" > "$O/$name.log" 2>&1 ) && echo "done $name $(date +%T)" || echo "FAILED $name"
}
export -f run_one; export O GPU
xargs -P "${JOBS:-5}" -L 1 bash -c 'run_one "$@"' _ < "$O/jobs.txt"
echo ALLDONE $(date)

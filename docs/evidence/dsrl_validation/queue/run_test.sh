#!/usr/bin/env bash
# DSRL test evaluation: 9 conditions (primary: 0cm, +-x/+-y 2cm; distance_4cm: +-x/+-y 4cm),
# 50 episodes each, eval seed 0, deterministic actor. One job = one policy x one condition
# (a fresh simulator per condition, as eval_dsrl.py does anyway), so an interrupted job costs
# at most 50 episodes. Jobs are listed in results/dsrl_test/jobs.txt as  name|policy|set|condition.
# Waits for a GPU with enough free memory, skips finished jobs, retries after OOM.
# usage: run_test.sh <shard> <nshards> [initial_delay_seconds]
set -u
SHARD=$1; NSHARDS=$2; DELAY=${3:-0}
MIN_FREE_MIB=5000
cd /home/sylee/steer-frozen-vla || exit 1
DIR=results/dsrl_test
sleep "$DELAY"

pick_gpu() {  # index of the GPU with the most free memory, empty if none has MIN_FREE_MIB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -n -r | head -1 | awk -F', ' -v m="$MIN_FREE_MIB" '$2 >= m {print $1}'
}

mapfile -t JOBS < "$DIR/jobs.txt"
for idx in "${!JOBS[@]}"; do
  [ $((idx % NSHARDS)) -eq "$SHARD" ] || continue
  IFS='|' read -r name policy set cond <<< "${JOBS[$idx]}"
  tag=${cond//+/p}; tag=${tag//-/m}
  mkdir -p "$DIR/$name"
  out=$DIR/$name/${set}_$tag.json
  log=$DIR/$name/${set}_$tag.log
  fails=0
  while :; do
    if [ -f "$out" ] && grep -q '"wall_seconds"' "$out"; then break; fi
    gpu=$(pick_gpu)
    if [ -z "$gpu" ]; then sleep 60; continue; fi
    rm -f "$out"
    source scripts/server_env.sh "$gpu"
    echo "[$(date '+%F %T')] $name $cond start gpu=$gpu"
    if "$PY" -B scripts/eval_dsrl.py --policy "$policy" --set "$set" --condition="$cond" \
         --out "$out" > "$log" 2>&1; then
      echo "[$(date '+%F %T')] $name $cond done: $(grep SUMMARY "$log" | sed 's/SUMMARY //' | tr '\n' ' ')"
      break
    fi
    if grep -qiE 'out of memory|CUDA error|CUDA-capable|cuDNN|EGL' "$log"; then
      echo "[$(date '+%F %T')] $name $cond GPU unavailable, retrying"
      sleep $((90 + RANDOM % 60)); continue
    fi
    fails=$((fails + 1)); cp "$log" "$log.fail$fails"
    echo "[$(date '+%F %T')] $name $cond FAILED attempt $fails (see $log.fail$fails)"
    if [ "$fails" -ge 3 ]; then echo "[$(date '+%F %T')] $name $cond GIVING UP"; break; fi
    sleep 30
  done
done
echo "[$(date '+%F %T')] shard $SHARD/$NSHARDS finished"

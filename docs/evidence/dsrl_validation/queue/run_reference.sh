#!/usr/bin/env bash
# Reference scores on the validation episodes (+-x / +-y 2cm, 10 episodes each, eval seed 1000)
# for a policy that is not a checkpoint: base (Frozen Base) or gaussian (untrained Gaussian latent).
# Waits for a GPU with enough free memory and retries after OOM.
# usage: run_reference.sh <base|gaussian> [initial_delay_seconds]
set -u
POLICY=$1; DELAY=${2:-0}
MIN_FREE_MIB=5000
cd /home/sylee/steer-frozen-vla || exit 1
DIR=results/dsrl_validation/reference
mkdir -p "$DIR"
out=$DIR/${POLICY}_val.json
log=$DIR/${POLICY}_val.log
sleep "$DELAY"
fails=0
while :; do
  if [ -f "$out" ] && grep -q '"wall_seconds"' "$out"; then break; fi
  gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -n -r | head -1 | awk -F', ' -v m="$MIN_FREE_MIB" '$2 >= m {print $1}')
  if [ -z "$gpu" ]; then sleep 60; continue; fi
  rm -f "$out"
  source scripts/server_env.sh "$gpu"
  echo "[$(date '+%F %T')] $POLICY start gpu=$gpu"
  if "$PY" -B scripts/eval_dsrl.py --policy "$POLICY" --set primary \
       --condition +x2 --condition=-x2 --condition +y2 --condition=-y2 \
       --episodes 10 --eval-seed 1000 --out "$out" > "$log" 2>&1; then
    echo "[$(date '+%F %T')] $POLICY done: $(grep SUMMARY "$log" | sed 's/SUMMARY //' | tr '\n' ' ')"
    break
  fi
  if grep -qiE 'out of memory|CUDA error|CUDA-capable|cuDNN|EGL' "$log"; then
    echo "[$(date '+%F %T')] $POLICY GPU unavailable, retrying"
    sleep $((90 + RANDOM % 60)); continue
  fi
  fails=$((fails + 1)); cp "$log" "$log.fail$fails"
  echo "[$(date '+%F %T')] $POLICY FAILED attempt $fails"
  if [ "$fails" -ge 3 ]; then echo "[$(date '+%F %T')] $POLICY GIVING UP"; break; fi
  sleep 30
done
echo "[$(date '+%F %T')] $POLICY finished"

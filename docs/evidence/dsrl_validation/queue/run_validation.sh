#!/usr/bin/env bash
# DSRL checkpoint validation. Applies the Residual run's pre-registered rule unchanged
# (configs/validation_rule.json on eval/residual-robustness):
#   candidates  ckpt at 2k..20k control steps, every 1k  -> ckpt_00000200 .. ckpt_00002000 (19)
#   evaluation  +-x / +-y 2cm, 10 episodes each, eval seed 1000, deterministic actor
# Waits for a GPU with enough free memory, skips finished checkpoints, retries after OOM.
# usage: run_validation.sh <seed> <shard> <nshards> [initial_delay_seconds]
set -u
SEED=$1; SHARD=$2; NSHARDS=$3; DELAY=${4:-0}
MIN_FREE_MIB=5000
cd /home/sylee/steer-frozen-vla || exit 1
RUN=results/dsrl_train_v1_s$SEED
mkdir -p "$RUN/validation"
sleep "$DELAY"

pick_gpu() {  # index of the GPU with the most free memory, empty if none has MIN_FREE_MIB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -n -r | head -1 | awk -F', ' -v m="$MIN_FREE_MIB" '$2 >= m {print $1}'
}

i=0
for n in $(seq 200 100 2000); do
  idx=$i; i=$((i + 1))
  [ $((idx % NSHARDS)) -eq "$SHARD" ] || continue
  name=$(printf 'ckpt_%08d' "$n")
  out=$RUN/validation/$name.json
  log=$RUN/validation/$name.log
  fails=0
  while :; do
    if [ -f "$out" ] && grep -q '"wall_seconds"' "$out"; then break; fi
    gpu=$(pick_gpu)
    if [ -z "$gpu" ]; then sleep 60; continue; fi
    rm -f "$out"
    source scripts/server_env.sh "$gpu"
    echo "[$(date '+%F %T')] s$SEED $name start gpu=$gpu"
    if "$PY" -B scripts/eval_dsrl.py --policy "$RUN/checkpoints/$name.pt" --set primary \
         --condition +x2 --condition=-x2 --condition +y2 --condition=-y2 \
         --episodes 10 --eval-seed 1000 --out "$out" > "$log" 2>&1; then
      echo "[$(date '+%F %T')] s$SEED $name done: $(grep SUMMARY "$log" | sed 's/SUMMARY //' | tr '\n' ' ')"
      break
    fi
    if grep -qiE 'out of memory|CUDA error|CUDA-capable|cuDNN|EGL' "$log"; then
      echo "[$(date '+%F %T')] s$SEED $name GPU unavailable, retrying"
      sleep $((90 + RANDOM % 60)); continue
    fi
    fails=$((fails + 1)); cp "$log" "$log.fail$fails"
    echo "[$(date '+%F %T')] s$SEED $name FAILED attempt $fails (see $log.fail$fails)"
    if [ "$fails" -ge 3 ]; then echo "[$(date '+%F %T')] s$SEED $name GIVING UP"; break; fi
    sleep 30
  done
done
echo "[$(date '+%F %T')] s$SEED shard $SHARD/$NSHARDS finished"

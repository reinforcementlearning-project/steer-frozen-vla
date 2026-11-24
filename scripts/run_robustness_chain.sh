#!/usr/bin/env bash
# Robustness protocol for one training seed of the fixed Residual-SAC candidate
# (6D residual, random +-x/+-y 2cm, SAC entropy coefficient fixed at 0.01):
#   wait for training -> pre-registered validation (configs/validation_rule.json)
#   -> test: final 20k and validation-selected checkpoints on 9 conditions x 50 ep (eval seed 0)
#   -> post-perturbation residual ablation of the selected checkpoint on the 8 perturbed conditions.
# Usage: scripts/run_robustness_chain.sh <seed> [parallel_jobs]
set -uo pipefail
cd "$(dirname "$0")/.."
seed="${1:?seed}"
JOBS="${2:-8}"
R="results/residual_6d_rand_fixa001_2cm_s${seed}_20k"
export OUT=results/residual_robustness/evals
mkdir -p "$OUT"
while [ -e "$R/process.pid" ] && kill -0 "$(cat "$R/process.pid")" 2>/dev/null; do sleep 30; done
[ -e "$R/summary.json" ] || { echo "TRAINING DID NOT FINISH: $R"; exit 1; }
echo "training done $(date)"
[ -e "$R/validation/selection.json" ] || scripts/validate_checkpoints.sh "$R" "$JOBS" || { echo "VALIDATION FAILED"; exit 1; }
SEL=$(python3 -c "import json;print(json.load(open('$R/validation/selection.json'))['selected_checkpoint'])")
echo "selected $SEL $(date)"
POLICIES="s${seed}_final --checkpoint $R/checkpoints/ckpt_00020000.pt
s${seed}_val --checkpoint $SEL" scripts/run_direction_eval.sh "$JOBS"
ONLY="py2 my2 px2 mx2 py4 my4 px4 mx4" \
POLICIES="s${seed}_val_postoff --checkpoint $SEL --residual-off-after-perturb" scripts/run_direction_eval.sh "$JOBS"
echo "ALLDONE $(date)"

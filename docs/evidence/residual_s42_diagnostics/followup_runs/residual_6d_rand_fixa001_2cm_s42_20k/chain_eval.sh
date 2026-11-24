#!/usr/bin/env bash
# train -> pre-registered validation/selection -> 9-condition test eval (final + selected)
set -uo pipefail
cd ~/rf/steer-frozen-vla
R=results/residual_6d_rand_fixa001_2cm_s42_20k
while kill -0 $(cat $R/process.pid) 2>/dev/null; do sleep 30; done
[ -e $R/summary.json ] || { echo "TRAINING DID NOT FINISH"; exit 1; }
echo "training done $(date)"
scripts/validate_checkpoints.sh $R 12 || { echo "VALIDATION FAILED"; exit 1; }
SEL=$(python3 -c "import json;print(json.load(open(\"$R/validation/selection.json\"))[\"selected_checkpoint\"])")
echo "selected $SEL $(date)"
POLICIES="fixa_final --checkpoint $R/checkpoints/ckpt_00020000.pt
fixa_val --checkpoint $SEL" scripts/run_direction_eval.sh 12
echo "ALLDONE $(date)"

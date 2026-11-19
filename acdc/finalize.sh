#!/usr/bin/env bash
# Re-run every evaluation on whatever predictions exist, rebuild results/summary.md, commit and push the results.
# Safe to run several times (each step overwrites its own outputs). Used as the last steps of the GPU queue.
set -u
cd "$(dirname "$0")/.."
UV=~/.local/bin/uv
SEG=""; for c in classical_lgbm unet2d unet25d unet2d_noaug unet2d_ce unet2d_notta unet2d_nopp; do
  [ -f results/seg/${c}_metrics.csv ] && SEG="$SEG $c"; done
SRC=""; for c in unet25d unet2d_noaug unet2d_ce; do [ -f results/seg/${c}_metrics.csv ] && SRC="$SRC $c"; done
echo "segmentation configs:$SEG | extra diagnosis sources:$SRC"
$UV run python -m acdc.evaluate_seg --cfgs $SEG
[ -n "$SRC" ] && $UV run python -m acdc.classify --sources $SRC
$UV run python -m acdc.evaluate_cls --seg_cfgs unet2d unet25d classical_lgbm $SRC
$UV run python -m acdc.summarize
git add results README.md acdc
git commit -q -m "Results update: evaluations re-run on all available models (${SEG# })" && git push -q origin main
echo "finalize done"

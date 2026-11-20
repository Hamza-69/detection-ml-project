#!/usr/bin/env bash
# All end-to-end CNN classifier runs: 5 seeds x 5 folds (skips runs that already finished).
cd "$(dirname "$0")/.."
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}  # uv on PATH, else the default install location
for s in 0 1 2 3 4; do for f in 0 1 2 3 4; do
  [ -f results/cls/cnn_runs/resnet18_s${s}_f${f}.csv ] && continue
  $UV run python -m acdc.train_cls --fold $f --seed $s 2>&1 | grep -v Warn | tail -n 2
done; done

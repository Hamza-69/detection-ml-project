#!/usr/bin/env bash
# Run train_cnn.py jobs one after another. Each argument is one job's flags, e.g.
#   ./run_queue.sh "--labels majority --seed 42" "--labels soft --seed 42"
# Logs go to /workspace/logs/<job>.log; a job that fails does not stop the queue.
cd "$(dirname "$0")"
for job in "$@"; do
  name=$(echo "$job" | tr -s ' -' '_' | sed 's/^_//')
  echo "[$(TZ=Asia/Beirut date +%H:%M)] start $job" >> /workspace/logs/queue.log
  uv run python train_cnn.py $job > "/workspace/logs/$name.log" 2>&1
  echo "[$(TZ=Asia/Beirut date +%H:%M)] end $job (exit $?)" >> /workspace/logs/queue.log
done

#!/usr/bin/env bash
# GPU job queue: runs the commands in $QUEUE one after another (the GPU is saturated by one job, so no parallel runs).
# Lines can be appended to the queue file while it runs; finished lines are recorded in $QUEUE.done.
# Usage: QUEUE=/path/queue.txt acdc/run_queue.sh     (run inside tmux)
set -u
QUEUE=${QUEUE:-$HOME/data/acdc/work/queue.txt}
LOGDIR=${LOGDIR:-$HOME/data/acdc/work/logs}
mkdir -p "$LOGDIR"; touch "$QUEUE.done"
cd "$(dirname "$0")/.."
while true; do
  next=$(grep -vxF -f "$QUEUE.done" "$QUEUE" | grep -v '^\s*#' | grep -v '^\s*$' | head -n 1)
  if [ -z "$next" ]; then sleep 30; continue; fi
  name=$(echo "$next" | tr -c 'A-Za-z0-9_' '_' | cut -c1-120)
  echo "$(date '+%F %T') START $next"
  bash -c "$next" > "$LOGDIR/$name.log" 2>&1
  echo "$(date '+%F %T') EXIT $? $next"
  echo "$next" >> "$QUEUE.done"
done

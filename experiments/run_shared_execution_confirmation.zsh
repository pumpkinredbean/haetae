#!/bin/zsh
set -euo pipefail

cd /Users/minkyu/workspace/haetae
export HF_HUB_OFFLINE=1

output=/Users/minkyu/workspace/haetae/evaluations/shared-execution-confirmation-v1

for process_index in 0 1 2 3 4 5; do
  marker="$output/process-${process_index}.meta.json"
  if [[ -f "$marker" ]]; then
    print -r -- "$(date -Iseconds) skip completed process ${process_index}"
  else
    print -r -- "$(date -Iseconds) start process ${process_index}"
    uv run python -u -m experiments.confirm_shared_execution measure \
      --out "$output" --process-index "$process_index"
    print -r -- "$(date -Iseconds) completed process ${process_index}"
  fi
done

if [[ -f "$output/summary.json" ]]; then
  print -r -- "$(date -Iseconds) skip completed summary"
else
  print -r -- "$(date -Iseconds) start summary"
  uv run python -u -m experiments.confirm_shared_execution summarize --out "$output"
  print -r -- "$(date -Iseconds) completed summary"
fi

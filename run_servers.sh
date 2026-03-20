#!/usr/bin/env bash
set -euo pipefail

# Logs next to where you run the script (current working dir)
LOGDIR="$(pwd)/logs"
mkdir -p "$LOGDIR"

start_one() {
  local gpus="$1" port="$2" model="$3" extra="$4" name="$5"
  local logfile="$LOGDIR/$name.out"

  echo "▶ Launch $name on GPU[$gpus] → :$port"
  echo "  Logs: $logfile"

  # Build command safely (avoids nested-quote issues)
  local -a cmd
  cmd=(python serve_llm.py --model "$model" --port "$port")
  # Split EXTRA args on spaces intentionally (you control the string)
  # shellcheck disable=SC2206
  cmd+=($extra)

  CUDA_VISIBLE_DEVICES="$gpus" nohup "${cmd[@]}" >"$logfile" 2>&1 &

  disown || true
}

start_one "0" "8001" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "--gpu_mem 0.9 --dtype bfloat16" \
  "server_Meta-Llama-3.1-8B-Instruct"

echo "All servers started. Logs → $LOGDIR/*.out"

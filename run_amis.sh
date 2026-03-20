#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere: resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Logs
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

RUN_NAME="amis_$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/${RUN_NAME}.out"

echo "▶ Starting AMIS run"
echo "  Run name: $RUN_NAME"
echo "  Log file: $LOGFILE"

# ---------------------------
# Config
# ---------------------------
OPTIM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TARGET_MODEL="gpt-4o-mini-2024-07-18"
SCORER_MODEL="gpt-4o-mini-2024-07-18"
SCORE_OPTIM_MODEL="gpt-4o-mini-2024-07-18"
ASR_MODEL="gpt-4o-mini-2024-07-18"

JAILBREAK_DATA_PATH="./data/advbench.json"
INITIAL_TEMPLATES_PATH=""
BEST_SEED_TEMPLATES_PATH=""
LOG_DIR="amis_log"

SCORING_ITERATIONS=5
SCORING_TOPK=5
SCORING_M=1

JAILBREAKING_ITERATIONS=5
JAILBREAKING_K=5
JAILBREAKING_M=5
JB_INHERIT_N=5

# HTTP_VLLM_URL="http://127.0.0.1:8001"
# HTTP_TIMEOUT=120

# ---------------------------
# Build command 
# ---------------------------
cmd=(
  python main.py
  --optim_model "$OPTIM_MODEL"
  --target_model "$TARGET_MODEL"
  --scorer_model "$SCORER_MODEL"
  --score_optim_model "$SCORE_OPTIM_MODEL"
  --asr_model "$ASR_MODEL"
  --jailbreak_data_path "$JAILBREAK_DATA_PATH"
  --scoring_iterations "$SCORING_ITERATIONS"
  --scoring_topK "$SCORING_TOPK"
  --scoring_M "$SCORING_M"
  --jailbreaking_iterations "$JAILBREAKING_ITERATIONS"
  --jailbreaking_K "$JAILBREAKING_K"
  --jailbreaking_M "$JAILBREAKING_M"
  --jb_inherit_n "$JB_INHERIT_N"
  --log_dir "$LOG_DIR"
#   --http_vllm_url "$HTTP_VLLM_URL"
#   --http_timeout "$HTTP_TIMEOUT"
)

# Optional args
if [[ -n "$INITIAL_TEMPLATES_PATH" ]]; then
  cmd+=(--initial_templates_path "$INITIAL_TEMPLATES_PATH")
fi

if [[ -n "$BEST_SEED_TEMPLATES_PATH" ]]; then
  cmd+=(--best_seed_templates_path "$BEST_SEED_TEMPLATES_PATH")
fi

# ---------------------------
# Run
# ---------------------------
echo "▶ Command:"
printf ' %q' "${cmd[@]}"
echo
echo

"${cmd[@]}" 2>&1 | tee "$LOGFILE"

echo
echo "Done. Logs saved to: $LOGFILE"
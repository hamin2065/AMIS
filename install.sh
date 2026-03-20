#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=AMIS
PYTHON_VER=3.10

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found"; exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install vllm fschat accelerate pandas openai anthropic spacy datasketch

python -m spacy download en_core_web_sm

python - <<'PY'
import torch
import vllm

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("vLLM:", vllm.__version__)
PY

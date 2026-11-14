#!/usr/bin/env bash
# Start DeepHash V5 with vLLM.
# Usage: ./vllm/serve.sh [b200|rtx5090] [extra vllm args...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-$(dirname "$HERE")}"
PROFILE="${1:-rtx5090}"; shift || true
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec vllm serve "$MODEL_DIR" --config "$HERE/$PROFILE.yaml" "$@"

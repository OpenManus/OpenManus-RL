#!/bin/bash

# Script to serve model with vLLM in OpenManus-RL Docker container

echo "========================================="
echo "Starting vLLM Model Server for OpenManus-RL"
echo "========================================="

# Enter the container and start vLLM
docker exec -it openmanus-rl bash -c '
# Set up environment variables for ROCm
export TRANSFORMERS_NO_TORCHVISION=1
export HF_HUB_DISABLE_TORCHVISION_IMPORT=1
export VLLM_USE_ROCM=1
export VLLM_PLATFORM=rocm
export LD_LIBRARY_PATH="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
):${LD_LIBRARY_PATH:-}:/opt/rocm/lib:/opt/rocm/lib64"

# Start vLLM server
echo "Starting vLLM server on port 8000..."
vllm serve /root/models/GiGPO-Qwen2.5-7B-Instruct-ALFWorld \
  --served-model-name qwen2.5-7b-alfworld \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.55 \
  --max-model-len 16384 \
  --enforce-eager \
  --device cuda \
  --host 0.0.0.0 --port 8000
'


#!/bin/bash

# --- Configuration ---
# Set this variable to "true" to use 8 GPUs (0-7) and tensor_parallel_size 8
# Set to "false" (or any other value) to use 2 GPUs (0,1) and tensor_parallel_size 2
use_all_gpu="false"

# --- Determine GPU settings based on the variable ---
if [ "$use_all_gpu" = "true" ]; then
    visible_devices="0,1,2,3,4,5,6,7"
    tensor_parallel_size=8
    echo "Configured to use 8 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
else
    visible_devices="6,7"
    tensor_parallel_size=2
    echo "Configured to use 4 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
fi

# MODEL_PATH="/home/user/models/Qwen/Qwen3-4B"
MODEL_PATH="/home/user/models/ZhipuAI/GLM-4-32B-0414"
# --- VLLM OpenAI API Server Command ---
# Set environment variables and run the vllm OpenAI-compatible server
echo "Starting VLLM OpenAI API server..."

CUDA_VISIBLE_DEVICES="$visible_devices" \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size "$tensor_parallel_size" \
    --host 0.0.0.0 \
    --port 8001 \
    --served-model-name agent-llm \
    --max-model-len 32768 
    # --enable-expert-parallel
    # --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
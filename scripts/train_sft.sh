#!/bin/bash
set -x

# Set default values for required parameters
default_nproc=2
default_save_path="./output"
default_use_lora=true
default_data_prefix="sft/data/"

# Parse command line arguments
nproc_per_node=${1:-$default_nproc}
save_path=${2:-$default_save_path}
use_lora=${3:-$default_use_lora}
data_prefix=${4:-$default_data_prefix}

# Shift the processed arguments
shift $(($# > 3 ? 4 : $#))

# Display usage information if needed
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $(basename $0) [nproc_per_node] [save_path] [use_lora] [data_prefix] [other_configs...]"
    echo "  nproc_per_node: Number of processes per node (default: $default_nproc)"
    echo "  save_path: Directory to save model and logs (default: $default_save_path)"
    echo "  use_lora: Whether to use LoRA for fine-tuning (true/false, default: $default_use_lora)"
    echo "  data_prefix: Path prefix for training data (default: $default_data_prefix)"
    exit 0
fi

# Create save directory if it doesn't exist
if [ ! -d "$save_path" ]; then
    mkdir -p "$save_path"
    echo "Created directory: $save_path"
fi

# Setup LoRA parameters if enabled
lora_config=""
if [ "$use_lora" = true ]; then
    lora_config="model.lora_rank=64 model.lora_alpha=32 model.target_modules=all-linear"
fi

# Extract environment type from data path for naming
env_type=$(basename $data_prefix)

# Generate a unique experiment name with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="finetune_${env_type}_${timestamp}"

# Run the training process
echo "Starting fine-tuning"
echo "Processes per node: $nproc_per_node"
echo "Save path: $save_path"
echo "Data path: $data_prefix"
echo "LoRA enabled: $use_lora"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${data_prefix}/train.parquet \
    data.val_files=${data_prefix}/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    optim.lr=1e-4 \
    data.train_batch_size=128 \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B \
    trainer.default_local_dir=$save_path \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    trainer.validate_before_training=True \
    model.enable_gradient_checkpointing=False \
    $lora_config \
    "$@" \
    2>&1 | tee $save_path/train.log

echo "Training completed. Logs saved to $save_path/train.log"
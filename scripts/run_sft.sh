#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

CONDA_BASE_DIR=$(conda info --base)
if [ -f "$CONDA_BASE_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
    conda activate verl
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment: verl"
        exit 1
    fi
else
    echo "Conda base profile script not found at $CONDA_BASE_DIR/etc/profile.d/conda.sh"
fi
export WANDB_API_KEY= # TODO: add your wandb api key here
wandb login

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

use_all_gpu="false"
# --- Determine GPU settings based on the variable ---
if [ "$use_all_gpu" = "true" ]; then
    visible_devices="0,1,2,3,4,5,6,7"
    tensor_parallel_size=8
    echo "Configured to use 8 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
else
    visible_devices="0,1,2,3"
    tensor_parallel_size=4
    echo "Configured to use 4 GPUs: CUDA_VISIBLE_DEVICES=$visible_devices, tensor_parallel_size=$tensor_parallel_size"
fi


# Set environment variables
CUDA_VISIBLE_DEVICES="$visible_devices" \
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=../data/train.parquet \
    data.val_files=../data/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=conversations \
    data.micro_batch_size=4 \
    model.partial_pretrain=/data1/models/Qwen/Qwen2.5-3B \ # TODO: add your model path here
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-4b \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
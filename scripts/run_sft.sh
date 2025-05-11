#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=OpenManus-RL/data/train_split.parquet \
    data.val_files=OpenManus-RL/data/test_split.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=prompt \
    data.micro_batch_size=4 \
    model.partial_pretrain=/data1/models/Qwen/Qwen3-4B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-3-4b \
    trainer.logger=['console'] \
    trainer.total_training_steps=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
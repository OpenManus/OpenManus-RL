BC_MODEL_PATH=
DATASET_FILE_PATH=PATH/TO/dpo_dataset.json
OUTPUT_PATH=

BETA=0.5
LR=5e-7

set -e

if [ ! -d ${OUTPUT_PATH} ]; then
    mkdir -p ${OUTPUT_PATH}
fi

torchrun --nproc_per_node=8 train_dpo_multiturn.py \
    --model_name_or_path ${BC_MODEL_PATH} \
    --ref_model_name_or_path ${BC_MODEL_PATH} \
    --trust_remote_code True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --num_train_epochs 3 \
    --beta ${BETA} \
    --learning_rate ${LR} \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --data_path ${DATASET_FILE_PATH} \
    --output_dir ${OUTPUT_PATH}/model \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 4096 \
    --max_prompt_length 1024 \
    --max_target_length 3072 \
    --gradient_checkpointing True

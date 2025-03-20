BC_MODEL_PATH=
DATASET_FILE_PATH=PATH/TO/Webshop_train.json
OUTPUT_PATH=
TESTSET_DATA_PATH=PATH/TO/Webshop_test.json

REWARD_GAP=0.1
EXPERT_THRESHOLD=0.9

BETA=0.5
LR=5e-7

ENV_SERVER_BASE_URL=

set -e

if [ ! -d ${OUTPUT_PATH} ]; then
    mkdir -p ${OUTPUT_PATH}
fi

# echo "Stage 0: evaluate the BC model"
# accelerate launch ../../utils/distributed_eval_task.py \
#     --model_path ${BC_MODEL_PATH} \
#     --output_file ${OUTPUT_PATH}/bc_webshop_eval.jsonl \
#     --inference_file ${TESTSET_DATA_PATH} \
#     --task_name "webshop" \
#     --eval_batch_size 1 \
#     --num_workers 8 \
#     --do_sample False \
#     --max_round 10 \
#     --env_server_base ${ENV_SERVER_BASE_URL} \
#     --data_len 200 \
#     --timeout 2400

echo "Stage 1: sample twice to get pair data"
accelerate launch ../../utils/distributed_eval_task.py \
    --model_path ${BC_MODEL_PATH} \
    --output_file ${OUTPUT_PATH}/dpo_webshop_inference_a.jsonl \
    --inference_file ${DATASET_FILE_PATH} \
    --task_name "webshop" \
    --eval_batch_size 1 \
    --num_workers 8 \
    --do_sample True \
    --temperature 1.0 \
    --max_round 10 \
    --env_server_base ${ENV_SERVER_BASE_URL} \
    --data_len 200 \
    --timeout 2400
accelerate launch ../../utils/distributed_eval_task.py \
    --model_path ${BC_MODEL_PATH} \
    --output_file ${OUTPUT_PATH}/dpo_webshop_inference_b.jsonl \
    --inference_file ${DATASET_FILE_PATH} \
    --task_name "webshop" \
    --eval_batch_size 1 \
    --num_workers 8 \
    --do_sample True \
    --temperature 1.0 \
    --max_round 10 \
    --env_server_base ${ENV_SERVER_BASE_URL} \
    --data_len 200 \
    --timeout 2400

echo "Stage 2: make the dpo dataset"
python -u make_dpo_dataset.py \
    --expert ${OUTPUT_PATH}/dpo_webshop_inference_a.jsonl \
    --experience ${OUTPUT_PATH}/dpo_webshop_inference_b.jsonl \
    --reward_gap ${REWARD_GAP} \
    --expert_threshold ${EXPERT_THRESHOLD} \
    --output ${OUTPUT_PATH}/dpo_webshop_dataset.jsonl

echo "Stage 3: train the dpo model"
torchrun --nproc_per_node=8 train_dpo_multiturn.py \
    --model_name_or_path ${BC_MODEL_PATH} \
    --ref_model_name_or_path ${BC_MODEL_PATH} \
    --trust_remote_code True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 5 \
    --num_train_epochs 3 \
    --beta ${BETA} \
    --learning_rate ${LR} \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 5 \
    --data_path ${OUTPUT_PATH}/dpo_webshop_dataset.jsonl \
    --output_dir ${OUTPUT_PATH}/dpo_webshop_model \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 4096 \
    --max_prompt_length 1024 \
    --max_target_length 3072 \
    --gradient_checkpointing True

echo "Stage 4: evaluate the dpo model"
accelerate launch ../../utils/distributed_eval_task.py \
    --model_path ${OUTPUT_PATH}/dpo_webshop_model \
    --output_file ${OUTPUT_PATH}/dpo_webshop_eval.jsonl \
    --inference_file ${TESTSET_DATA_PATH} \
    --task_name "webshop" \
    --eval_batch_size 1 \
    --num_workers 8 \
    --do_sample False \
    --max_round 10 \
    --env_server_base ${ENV_SERVER_BASE_URL} \
    --data_len 200 \
    --timeout 2400

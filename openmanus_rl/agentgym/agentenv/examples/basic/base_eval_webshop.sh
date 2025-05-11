# Evaluation args
model_path="/data1/models/openmanus_rl/Qwen/Qwen3-3b-sft/global_step_1"
inference_file="/home/user/muxin/OpenManus-RL/data/webshop/webshop_inference.json"
output_file="/data1/models/openmanus_rl/Qwen/Qwen3-3b-sft/output/qwen2.5-3b-webshop.log"
task_name="webshop"
seed="42"

# environment parameters
max_round="6"
env_server_base="http://127.0.0.1:36001"

python -u base_eval_template.py \
        --model_path "${model_path}" \
        --inference_file "${inference_file}" \
        --output_file "${output_file}" \
        --task_name "${task_name}" \
        --seed "${seed}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}"

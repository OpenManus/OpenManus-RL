# Evaluation args
model_path=""
inference_file=""
output_file=""
task_name=""
seed="42"

# environment parameters
max_round=""
env_server_base=""

python -u base_eval_template.py \
        --model_path "${model_path}" \
        --inference_file "${inference_file}" \
        --output_file "${output_file}" \
        --task_name "${task_name}" \
        --seed "${seed}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}"

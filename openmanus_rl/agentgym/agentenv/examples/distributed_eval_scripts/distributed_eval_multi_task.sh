exp_name="eval_multi_task"

# accelerator config
num_processes='8'
main_process_port='8897'
config_file=""

# training arguments
model_path="THUDM/agentlm-7b"

eval_batch_size="1"
num_workers="8"
seed="42"

# environment parameters
data_len="200"
timeout="2400"

# eval config
do_sample="False"
temperature="1.0"
sample_num="1"

# eval
task_list=(
    "webshop"
    "alfworld"
    "textcraft"
    "babyai"
)

# eval parameters
test_file_list=(
    "PATH/TO/THE/webshop_test.json"
    "PATH/TO/THE/alfworld_test.json"
    "PATH/TO/THE/textcraft_test.json"
    "PATH/TO/THE/babyai_test.json"
)

# inference parameters
max_round_list=(
    "10"
    "30"
    "20"
    "20"
)
env_server_base_list=(
    "http://127.0.0.1:59311"
    "http://127.0.0.1:59220"
    "http://127.0.0.1:59317"
    "http://127.0.0.1:59330"
)

# step2: eval on test dataset
for index in "${!task_list[@]}"; do
    cur_task=${task_list[$index]}
    cur_test_file="${test_file_list[$index]}"
    cur_max_round=${max_round_list[$index]}
    cur_env_server_base=${env_server_base_list[$index]}
    
    echo "${cur_eval_output_file}"
    echo "${cur_task}"
    
    accelerate launch \
            --config_file "${config_file}" \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
        ../../utils/distributed_eval_task.py \
            --model_path "${model_path}" \
            --output_file "${cur_eval_output_file}" \
            --inference_file "${cur_test_file}" \
            --task_name "${cur_task}" \
            --eval_batch_size "${eval_batch_size}" \
            --num_workers "${num_workers}" \
            --seed "${seed}" \
            --do_sample "${do_sample}" \
            --max_round "${cur_max_round}" \
            --env_server_base "${cur_env_server_base}" \
            --data_len "${data_len}" \
            --timeout "${timeout}"
done

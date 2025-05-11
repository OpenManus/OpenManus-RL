exp_name="eval_webshop"
inference_file='/home/user/muxin/OpenManus-RL/data/webshop/webshop_inference.json' # Path to the trainset file which contains idxs for the task.

num_processes='8'
main_process_port='8877'
weight_decay="0"

### Default variables
task_name="webshop" # change this to evaluate on a different task
output_dir="/data1/models/openmanus_rl/Qwen/Qwen3-3b-sft/output"

# agent model
#model_path="/data1/models/openmanus_rl/Qwen/Qwen3-3b-sft/global_step_1"
model_path="/data1/models/Qwen/Qwen2.5-3B"
eval_batch_size="1"
num_workers="8"
seed="42"
do_sample="False"
temperature="1.0"

max_round="6"
env_server_base="http://127.0.0.1:36001" # Set this to the base url of the EnvServer.
timeout="2400"


#########
mkdir -p "${output_dir}"
export PYTHONPATH=/home/user/muxin/OpenManus-RL/openmanus_rl/agentgym/agentenv:$PYTHONPATH # You need to modify this as your agentgym/agentenv absolute path

accelerate launch \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    ../../utils/distributed_eval_task.py \
        --model_path "${model_path}" \
        --output_file "${output_dir}/inference.jsonl" \
        --inference_file "${inference_file}" \
        --task_name "${task_name}" \
        --eval_batch_size "${eval_batch_size}" \
        --num_workers "${num_workers}" \
        --seed "${seed}" \
        --do_sample "${do_sample}" \
        --temperature "${temperature}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}" \
        --data_len 200 \
        --timeout "${timeout}"

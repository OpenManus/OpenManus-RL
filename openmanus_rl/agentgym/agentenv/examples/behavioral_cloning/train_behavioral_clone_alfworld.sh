exp_name="behavioral_clone_alfworld_1000"

n_epochs='1'

# accelerator config
num_processes='8'
main_process_port='8895'
config_file=""

# training arguments
train_file='PATH/TO/alfworld_train_1000.json'
model_train_path="meta-llama/Llama-2-7b-chat-hf"
model_save_path="outputs/${exp_name}/"

batch_size="2"
eval_batch_size="1"
gradient_accumulation_steps="2"
max_input_length="4096"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
evaluating_epoch_freq="100"
saving_epoch_freq="1"
logging_step_freq="5"

# wandb config
wandb_log="True"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="200"
timeout="2400"

# eval
task_list=("webshop" "alfworld" "textcraft" "sciworld")
# eval parameters
test_file_list=("PATH/TO/webshop_test.json" "PATH/TO/alfworld_test.json" "PATH/TO/textcraft_test.json" "PATH/TO/sciworld_test_small.json")
do_sample="False"
temperature="1.0"
sample_num="2"
max_round_list=("6" "30" "20" "30")
env_server_base_list=("http://127.0.0.1:36004" "http://127.0.0.1:36002" "http://127.0.0.1:36008" "http://127.0.0.1:36010")

mkdir -p "${model_save_path}"
# step1: train
accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    train_behavioral_clone.py \
        --train_file "${train_file}" \
        --inference_file "${test_file_list[1]}" \
        --test_file "${test_file_list[1]}" \
        --model_train_path "${model_train_path}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[1]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[1]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --env_server_base "${env_server_base_list[1]}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"

# step2: eval on test dataset
cur_task=${task_list[1]}
test_file=${test_file_list[1]}
max_round=${max_round_list[1]}
env_server_base=${env_server_base_list[1]}
eval_output_file="${model_save_path}/eval_${cur_task}.jsonl"

accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    ../../utils/distributed_eval_task.py \
        --model_path "${model_save_path}/train_epoch_${n_epochs}" \
        --output_file "${eval_output_file}" \
        --inference_file "${test_file}" \
        --task_name "${cur_task}" \
        --eval_batch_size "${eval_batch_size}" \
        --num_workers "${num_workers}" \
        --seed "${seed}" \
        --do_sample "${do_sample}" \
        --max_round "${max_round}" \
        --env_server_base "${env_server_base}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"

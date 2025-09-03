#!/bin/bash
# Modular training script for ALFWorld with module-specific reward model evaluation

set -x
ENGINE=${1:-vllm}
TRAIN_MODULAR=${2:-false}  # New parameter for modular training
REWARD_MODEL_PATH=${3:-/home/user/models/Qwen/Qwen3-4B}  # Path to reward model
REWARD_MODEL_PORT=${4:-8100}  # Port for reward model server

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=
export WANDB_BASE_URL=https://api.bandw.top

visible_devices="1,2,3,4"
export CUDA_VISIBLE_DEVICES="$visible_devices"

train_data_size=128
val_data_size=128

# Start reward model server if modular training is enabled
if [ "$TRAIN_MODULAR" = true ]; then
    echo "Starting reward model server for modular training..."
    
    # Kill any existing process on the port
    lsof -ti:$REWARD_MODEL_PORT | xargs kill -9 2>/dev/null || true
    
    # Start the reward model server in background
    python3 /root/kunlunz/OpenManus-RL/scripts/host_reward_model.py \
        --model-path $REWARD_MODEL_PATH \
        --port $REWARD_MODEL_PORT \
        --gpu-memory 0.2 &
    
    REWARD_MODEL_PID=$!
    echo "Reward model server started with PID: $REWARD_MODEL_PID"
    
    # Wait for server to be ready
    sleep 10
    
    # Check if server is running
    if ! curl -s http://localhost:$REWARD_MODEL_PORT/health > /dev/null; then
        echo "Error: Reward model server failed to start"
        exit 1
    fi
    
    echo "Reward model server is ready"
fi

# Main training command with modular training support
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=/data1/user/muxin/verl-agent/text/train.parquet \
    data.val_files=/data1/user/muxin/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/home/user/models/Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/home/user/models/Qwen/Qwen3-4B \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=128 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='openmanus-rl_ppo_alfworld_modular' \
    trainer.experiment_name='ppo_qwen3_4b_modular' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.train_modular=$TRAIN_MODULAR \
    trainer.reward_model_url=http://localhost:$REWARD_MODEL_PORT \
    trainer.env_reward_weight=0.5 \
    trainer.model_reward_weight=0.5 \
    trainer.module_iterations_per_epoch=4 \
    trainer.random_seed=42 $@

# Cleanup: Kill reward model server when training finishes
if [ "$TRAIN_MODULAR" = true ]; then
    echo "Stopping reward model server..."
    kill $REWARD_MODEL_PID 2>/dev/null || true
    echo "Reward model server stopped"
fi

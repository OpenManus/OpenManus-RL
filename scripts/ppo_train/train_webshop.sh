export HIP_VISIBLE_DEVICES=0

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 # Patch with https://github.com/ray-project/ray/pull/52794

# WANDB Configuration
# Set your WANDB API key here or in your environment
export WANDB_API_KEY="6105114af5891671eb71ca545c458d6df825dc32"  # Uncomment and add your key
export WANDB_PROJECT="openmanus-rl_ppo_webshop"  # Project name for WANDB
export WANDB_ENTITY="scottctd"  # Your WANDB username or team name
export WANDB_MODE="online"  # Options: online, offline, disabled
export WANDB_TAGS="ppo,webshop,qwen2.5"  # Tags for this run
export WANDB_RUN_NAME="ppo_webshop_$(date +%Y%m%d_%H%M%S)"  # Auto-generated run name with timestamp

# Optional WANDB settings
export WANDB_SILENT="false"  # Set to true to suppress WANDB output
export WANDB_DIR="./wandb"  # Directory for WANDB logs
# export WANDB_RESUME="allow"  # Options: allow, must, never, auto
# export WANDB_RUN_ID="specific_run_id"  # Use to resume a specific run

train_data_size=1
val_data_size=1

ENGINE=vllm #sglang

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=data/dummy/empty.parquet \
    data.val_files=data/dummy/empty.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.use_invalid_action_penalty=True \
    +actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    +env.env_name=Webshop \
    +env.seed=0 \
    +env.max_steps=9 \
    +env.rollout.n=4 \
    +env.webshop.use_small=False \
    +env.webshop.human_goals=False \
    +env.history_length=4 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='openmanus-rl_ppo_webshop' \
    trainer.experiment_name='ppo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@


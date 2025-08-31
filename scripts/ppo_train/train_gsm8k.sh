YOUR_PROJECT_NAME=r1-verl-ppo-upstream
YOUR_RUN_NAME=r1-training_ppo-upstream
# export HYDRA_FULL_ERROR=1

export HIP_VISIBLE_DEVICES=0

# [ray] < 2.45.0
#export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

# [ray] >= 2.45.0
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 # Patch with https://github.com/ray-project/ray/pull/52794

GPUS_PER_NODE=1
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
python3 verl/examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
python3 -c "import transformers; transformers.pipeline('text-generation', model='$MODEL_PATH')"
ENGINE=vllm #sglang

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=data/gsm8k/train.parquet \
 data.val_files=data/gsm8k/test.parquet \
 data.train_batch_size=256 \
 data.val_batch_size=1312 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL_PATH \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.name=$ENGINE \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL_PATH \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.project_name=$YOUR_PROJECT_NAME \
 trainer.experiment_name=$YOUR_RUN_NAME \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=$GPUS_PER_NODE \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 #2>&1 | tee verl_demo.log
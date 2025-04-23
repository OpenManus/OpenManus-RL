#!/bin/bash

# --- Configuration (defaults, can be overridden via env vars) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WAND_PROJECT=${WAND_PROJECT:-'OpenManus-rl'}
export BASE_MODEL=${BASE_MODEL:-'meta-llama/Llama-3.2-3B'}
AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'}
AGENTGYM_SQL_BIRD_PATH=${AGENTGYM_SQL_BIRD_PATH:-} # Used only for sqlgym

# --- Argument Parsing ---
usage() {
    echo "Usage: $0 --env_name <environment_name> [--port <port>] [--data_dir <path>] [--exp_name_suffix <suffix>]"
    echo "Supported env_names: webshop, webarena, maze, wordle, alfworld, sciworld, babyai, textcraft, weather, movie, academia, todo, sheet, sqlgym"
    echo "Assumes dedicated conda environments like 'agentenv-webshop' are already created and set up."
    exit 1
}

AGENTGYM_ENV_NAME="webshop" # Default environment
AGENTGYM_PORT_OVERRIDE=""
DATA_DIR_OVERRIDE=""
EXP_NAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env_name)
            AGENTGYM_ENV_NAME="$2"; shift; shift;;
        --port)
            AGENTGYM_PORT_OVERRIDE="$2"; shift; shift;;
        --data_dir)
            DATA_DIR_OVERRIDE="$2"; shift; shift;;
        --exp_name_suffix)
            EXP_NAME_SUFFIX="_$2"; shift; shift;;
        *)
            echo "Unknown option: $1"; usage;;
    esac
done

if [ -z "$AGENTGYM_ENV_NAME" ]; then
    echo "Error: --env_name is required."; usage
fi

# --- Determine Base Environment (where verl runs) ---
BASE_CONDA_ENV=${CONDA_DEFAULT_ENV:-openmanus-rl}
echo "[Info] Detected base conda environment: $BASE_CONDA_ENV"
echo "[Info] Verl trainer will run in this environment."

# --- Environment Specific Setup (Determine LAUNCH_CMD, DEFAULT_PORT, URL_PATH) ---
LAUNCH_CMD=""
DEFAULT_PORT=""
URL_PATH=""
# 不再使用Python -m 模块方式
# MODULE_LAUNCH_NAME=""

# 设置默认主机为0.0.0.0以允许外部访问
AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'}

case $AGENTGYM_ENV_NAME in
    webshop)
        LAUNCH_CMD="webshop --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    webarena)
        LAUNCH_CMD="webarena --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=8000;;
    maze)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001; URL_PATH="/maze/";;
    wordle)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001; URL_PATH="/wordle/";;
    alfworld)
        LAUNCH_CMD="alfworld --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    sciworld)
        LAUNCH_CMD="sciworld --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    babyai)
        LAUNCH_CMD="babyai --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    textcraft)
        LAUNCH_CMD="textcraft --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    weather|movie|academia|todo|sheet)
        LAUNCH_CMD="\$AGENTGYM_ENV_NAME --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=8000;;
    sqlgym)
        if [ -z "$AGENTGYM_SQL_BIRD_PATH" ]; then echo "Error: AGENTGYM_SQL_BIRD_PATH must be set for sqlgym."; exit 1; fi
        LAUNCH_CMD="AGENTENV_SQLGYM_BIRD_PATH=$AGENTGYM_SQL_BIRD_PATH sqlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36002;;
    *)
        echo "Error: Unsupported environment name '$AGENTGYM_ENV_NAME'"; usage;;
esac

# --- Start AgentGym Server in its Dedicated Environment ---
TARGET_ENV_NAME="agentenv-${AGENTGYM_ENV_NAME}"
AGENTGYM_PID=""

# Check if target env exists
if ! conda env list | grep -Eq "^${TARGET_ENV_NAME}\s"; then
    echo "[Error] Dedicated environment '$TARGET_ENV_NAME' not found. Please create it first."
    exit 1
fi

# Prepare Launch Command (Prefer python -m style if defined)
export AGENTGYM_PORT=${AGENTGYM_PORT_OVERRIDE:-$DEFAULT_PORT}

# 直接使用命令行工具方式启动
FINAL_LAUNCH_CMD=$(eval echo $LAUNCH_CMD) # Substitute $AGENTGYM_PORT

echo -e "\\n[Server] Starting AgentGym server for ${AGENTGYM_ENV_NAME} in env '$TARGET_ENV_NAME'..."
echo "[Server] Host: ${AGENTGYM_HOST}, Port: ${AGENTGYM_PORT}"
echo "[Server] Launch command: $FINAL_LAUNCH_CMD"

# Create logs directory
mkdir -p logs

# Run server in background using conda run in the target environment
LOG_FILE="logs/${TARGET_ENV_NAME}_server.log"
echo "[Server] Logging to $LOG_FILE"

# 简化启动命令，直接使用conda run启动服务器
conda run -n "$TARGET_ENV_NAME" $FINAL_LAUNCH_CMD > "$LOG_FILE" 2>&1 &
AGENTGYM_PID=$!

# Check if PID was obtained
if [ -z "$AGENTGYM_PID" ]; then
    echo "[Error] Failed to get PID for AgentGym server launch command."
    exit 1
fi
echo "[Server] AgentGym server launched in '$TARGET_ENV_NAME' (PID: $AGENTGYM_PID)."

# --- Wait and Check Server ---
echo "[Server] Waiting for AgentGym server (PID: $AGENTGYM_PID) to initialize..."
sleep 15 # Adjust sleep time if needed

# Check if the process with the captured PID is still running
if ! kill -0 $AGENTGYM_PID > /dev/null 2>&1; then
    echo "[Error] AgentGym server (PID: $AGENTGYM_PID) failed to start or exited prematurely."
    echo "[Error] Check server log: $LOG_FILE"
    exit 1
fi
echo "[Server] AgentGym server appears to be running (PID: $AGENTGYM_PID)."

# Setup trap to kill the server process on script exit/interrupt
trap "echo '[Cleanup] Stopping AgentGym server (PID: $AGENTGYM_PID)...'; kill $AGENTGYM_PID 2>/dev/null || echo '[Cleanup] Server already stopped.'" EXIT

# --- Data and Experiment Naming ---
export DATA_DIR=${DATA_DIR_OVERRIDE:-"data/$AGENTGYM_ENV_NAME"} # Default data dir based on env name
export EXPERIMENT_NAME="OpenManus-rl-ppo-${BASE_MODEL##*/}-${AGENTGYM_ENV_NAME}${EXP_NAME_SUFFIX}"

# --- Run PPO Training in Base Environment ---
echo -e "\\n[Trainer] Running PPO training in base environment '$BASE_CONDA_ENV'..."
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# Construct server base URL, adding path if needed
AGENTGYM_SERVER_BASE="http://$AGENTGYM_HOST"
if [ -n "$URL_PATH" ]; then
    AGENTGYM_SERVER_BASE="$AGENTGYM_SERVER_BASE$URL_PATH"
fi

echo "[Trainer] Using Data Directory: $DATA_DIR"
echo "[Trainer] Experiment Name: $EXPERIMENT_NAME"
echo "[Trainer] AgentGym Base URL: $AGENTGYM_SERVER_BASE:$AGENTGYM_PORT"

# Check if train/test files exist
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[Warning] Train file not found at $TRAIN_FILE. Ensure data generation script was run for $AGENTGYM_ENV_NAME."
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "[Warning] Test file not found at $TEST_FILE. Ensure data generation script was run for $AGENTGYM_ENV_NAME."
fi

TRAINER_LOG_FILE="logs/${EXPERIMENT_NAME}.log"
echo "[Trainer] Logging trainer output to $TRAINER_LOG_FILE"

# 正确方式激活conda环境（使用source）
echo "[Trainer] Activating base environment for training..."
# 获取conda base路径并确保conda.sh可用
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate openmanus-rl

# 检查并安装必要的依赖
echo "[Trainer] Checking and installing required dependencies..."
if ! python -c "import tensordict" &>/dev/null; then
    echo "[Trainer] Installing missing dependency: tensordict"
    pip install tensordict
fi

echo "[Trainer] Starting PPO training..."
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.env_name=$AGENTGYM_ENV_NAME \
    data.env_server_base=$AGENTGYM_SERVER_BASE \
    data.env_port=$AGENTGYM_PORT \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    algorithm.reward_score_fn=agentgym \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=305 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=2 \
    2>&1 | tee "$TRAINER_LOG_FILE" # Log trainer output

TRAINER_EXIT_CODE=$?

echo "PPO training finished with exit code $TRAINER_EXIT_CODE."

# Cleanup is handled by the trap

exit $TRAINER_EXIT_CODE
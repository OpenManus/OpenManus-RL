#!/bin/bash

# --- Configuration (defaults, can be overridden via env vars) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WAND_PROJECT=${WAND_PROJECT:-'OpenManus-rl'}
export BASE_MODEL=${BASE_MODEL:-'Qwen/Qwen2.5-3B'}
AGENTGYM_HOST=${AGENTGYM_HOST:-'127.0.0.1'}
AGENTGYM_SQL_BIRD_PATH=${AGENTGYM_SQL_BIRD_PATH:-} # Used only for sqlgym

# --- Argument Parsing ---
usage() {
    echo "Usage: $0 --env_name <environment_name> [--port <port>] [--data_dir <path>] [--exp_name_suffix <suffix>]"
    echo "Supported env_names: webshop, webarena, maze, wordle, alfworld, sciworld, babyai, textcraft, weather, movie, academia, todo, sheet, sqlgym"
    exit 1
}

AGENTGYM_ENV_NAME="webshop"
AGENTGYM_PORT_OVERRIDE=""
DATA_DIR_OVERRIDE=""
EXP_NAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env_name)
            AGENTGYM_ENV_NAME="$2"
            shift; shift;;
        --port)
            AGENTGYM_PORT_OVERRIDE="$2"
            shift; shift;;
        --data_dir)
            DATA_DIR_OVERRIDE="$2"
            shift; shift;;
        --exp_name_suffix)
            EXP_NAME_SUFFIX="_$2"
            shift; shift;;
        *)
            echo "Unknown option: $1"
            usage;;
    esac
done

if [ -z "$AGENTGYM_ENV_NAME" ]; then
    echo "Error: --env_name is required."
    usage
fi

# --- Environment Specific Setup ---
LAUNCH_CMD=""
DEFAULT_PORT=""
URL_PATH=""

case $AGENTGYM_ENV_NAME in
    webshop)
        LAUNCH_CMD="webshop --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001;;
    webarena)
        LAUNCH_CMD="webarena --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=8000;;
    maze)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001
        URL_PATH="/maze/";;
    wordle)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36001
        URL_PATH="/wordle/";;
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
        if [ -z "$AGENTGYM_SQL_BIRD_PATH" ]; then
            echo "Error: AGENTGYM_SQL_BIRD_PATH environment variable must be set for sqlgym."
            exit 1
        fi
        LAUNCH_CMD="AGENTENV_SQLGYM_BIRD_PATH=$AGENTGYM_SQL_BIRD_PATH sqlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_PORT=36002;;
    *)
        echo "Error: Unsupported environment name '$AGENTGYM_ENV_NAME'"
        usage;;
esac

# --- Environment Dependency Installation (in‑place) ---
ENV_SETUP_DIR="openmanus_rl/agentgym/agentenv-${AGENTGYM_ENV_NAME}"
if [ -d "$ENV_SETUP_DIR" ]; then
    echo -e "\n[Setup] Preparing AgentGym environment '$AGENTGYM_ENV_NAME' from $ENV_SETUP_DIR ..."
    pushd "$ENV_SETUP_DIR" > /dev/null || { echo "Failed to enter $ENV_SETUP_DIR"; exit 1; }

    # install requirements
    if [ -f "requirements.txt" ]; then
        echo "[Setup] Installing Python requirements ..."
        pip install --no-cache-dir -r requirements.txt
    fi
    # run setup.sh
    if [ -f "setup.sh" ]; then
        echo "[Setup] Running setup.sh ..."
        bash setup.sh
    fi
    # editable install
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        echo "[Setup] Installing environment package (editable) ..."
        pip install -e .
    fi
    popd > /dev/null
    echo "[Setup] Environment '$AGENTGYM_ENV_NAME' ready."
else
    echo "[Setup] WARNING: $ENV_SETUP_DIR not found; skipping env‑specific installation."
fi

export AGENTGYM_PORT=${AGENTGYM_PORT_OVERRIDE:-$DEFAULT_PORT}
FINAL_LAUNCH_CMD=$(eval echo $LAUNCH_CMD)

# --- Data & Experiment Naming ---
export DATA_DIR=${DATA_DIR_OVERRIDE:-"data/$AGENTGYM_ENV_NAME"}
export EXPERIMENT_NAME="OpenManus-rl-grpo-${BASE_MODEL##*/}-${AGENTGYM_ENV_NAME}${EXP_NAME_SUFFIX}"

# --- Start AgentGym Server ---
echo "Starting AgentGym server for ${AGENTGYM_ENV_NAME} on ${AGENTGYM_HOST}:${AGENTGYM_PORT} ..."
echo "Launch command: $FINAL_LAUNCH_CMD"
$FINAL_LAUNCH_CMD &
AGENTGYM_PID=$!
echo "AgentGym server started with PID: $AGENTGYM_PID"

sleep 10
if ! kill -0 $AGENTGYM_PID > /dev/null 2>&1; then
    echo "AgentGym server failed to start."
    exit 1
fi
trap "echo 'Stopping AgentGym server (PID: $AGENTGYM_PID)...'; kill $AGENTGYM_PID" EXIT

# --- Run GRPO Training ---
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

AGENTGYM_SERVER_BASE="http://$AGENTGYM_HOST"
if [ -n "$URL_PATH" ]; then
    AGENTGYM_SERVER_BASE="$AGENTGYM_SERVER_BASE$URL_PATH"
fi

echo "Using Data Directory: $DATA_DIR"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "AgentGym Base URL: $AGENTGYM_SERVER_BASE:$AGENTGYM_PORT"

TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "Warning: Train file not found at $TRAIN_FILE"
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "Warning: Test file not found at $TEST_FILE"
fi

mkdir -p logs

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
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.use_kl_loss=true \
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
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
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
    2>&1 | tee logs/${EXPERIMENT_NAME}.log

# Clean up server
kill $AGENTGYM_PID
wait $AGENTGYM_PID 2>/dev/null

echo "AgentGym server stopped."
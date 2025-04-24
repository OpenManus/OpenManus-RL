#!/bin/bash

# --- Configuration (defaults, can be overridden via env vars) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
WAND_PROJECT=${WAND_PROJECT:-'OpenManus-rl'}
export BASE_MODEL=${BASE_MODEL:-'meta-llama/Llama-3.2-3B'}
AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'} # Default to 0.0.0.0 for external access
AGENTGYM_SQL_BIRD_PATH=${AGENTGYM_SQL_BIRD_PATH:-} # Used only for sqlgym

# --- Argument Parsing ---
usage() {
    echo "Usage: $0 --env_name <environment_name> [--num_servers <N>] [--base_port <port>] [--data_dir <path>] [--exp_name_suffix <suffix>]"
    echo "Supported env_names: webshop, webarena, maze, wordle, alfworld, sciworld, babyai, textcraft, weather, movie, academia, todo, sheet, sqlgym"
    echo "  --num_servers: Number of parallel AgentGym servers to launch (default: 1)."
    echo "  --base_port: Starting port number for servers (default varies by env)."
    echo "Assumes dedicated conda environments like 'agentenv-webshop' are already created and set up."
    exit 1
}

AGENTGYM_ENV_NAME="webshop" # Default environment
NUM_SERVERS=1 # Default number of servers
BASE_PORT_OVERRIDE=""
DATA_DIR_OVERRIDE=""
EXP_NAME_SUFFIX=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env_name)
            AGENTGYM_ENV_NAME="$2"; shift; shift;;
        --num_servers)
            NUM_SERVERS="$2"; shift; shift;;
        --base_port) # Changed from --port to --base_port
            BASE_PORT_OVERRIDE="$2"; shift; shift;;
        --data_dir)
            DATA_DIR_OVERRIDE="$2"; shift; shift;;
        --exp_name_suffix)
            EXP_NAME_SUFFIX="_$2"; shift; shift;;
        *)
            echo "Unknown option: $1"; usage;;
    esac
done

if ! [[ "$NUM_SERVERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num_servers must be a positive integer."
    usage
fi

if [ -z "$AGENTGYM_ENV_NAME" ]; then
    echo "Error: --env_name is required."; usage
fi

# --- Determine Base Environment (where verl runs) ---
BASE_CONDA_ENV=${CONDA_DEFAULT_ENV:-openmanus-rl}
echo "[Info] Detected base conda environment: $BASE_CONDA_ENV"
echo "[Info] Verl trainer will run in this environment."


# --- Environment Specific Setup (Determine LAUNCH_CMD, DEFAULT_BASE_PORT, URL_PATH) ---

LAUNCH_CMD=""
DEFAULT_BASE_PORT="" # Renamed from DEFAULT_PORT
URL_PATH=""
# MODULE_LAUNCH_NAME=""

AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'}

case $AGENTGYM_ENV_NAME in
    webshop)
        LAUNCH_CMD="webshop --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001;;
    webarena)
        LAUNCH_CMD="webarena --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=8000;;
    maze)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001; URL_PATH="/maze/";;
    wordle)
        LAUNCH_CMD="lmrlgym --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001; URL_PATH="/wordle/";;
    alfworld)
        LAUNCH_CMD="alfworld --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001;;
    sciworld)
        LAUNCH_CMD="sciworld --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001;;
    babyai)
        LAUNCH_CMD="babyai --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001;;
    textcraft)
        LAUNCH_CMD="textcraft --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36001;;
    weather|movie|academia|todo|sheet)
        LAUNCH_CMD="\\\$AGENTGYM_ENV_NAME --host $AGENTGYM_HOST --port \\\$AGENTGYM_PORT" # Escaped env name var
        DEFAULT_BASE_PORT=8000;;
    sqlgym)
        if [ -z "$AGENTGYM_SQL_BIRD_PATH" ]; then echo "Error: AGENTGYM_SQL_BIRD_PATH must be set for sqlgym."; exit 1; fi
        LAUNCH_CMD="AGENTENV_SQLGYM_BIRD_PATH=$AGENTGYM_SQL_BIRD_PATH sqlgym --host $AGENTGYM_HOST --port \\\$AGENTGYM_PORT"
        DEFAULT_BASE_PORT=36002;;
    *)
        echo "Error: Unsupported environment name '$AGENTGYM_ENV_NAME'"; usage;;
esac

# --- Start AgentGym Servers in Dedicated Environment ---
TARGET_ENV_NAME="agentenv-${AGENTGYM_ENV_NAME}"
AGENTGYM_PIDS=() # Array to store PIDs
AGENTGYM_PORTS=() # Array to store ports

# Check if target env exists
if ! conda env list | grep -Eq "^${TARGET_ENV_NAME}\\s"; then
    echo "[Error] Dedicated environment '$TARGET_ENV_NAME' not found. Please create it first."
    exit 1
fi

# Determine base port
AGENTGYM_BASE_PORT=${BASE_PORT_OVERRIDE:-$DEFAULT_BASE_PORT}

echo -e "\\n[Server] Starting $NUM_SERVERS AgentGym server(s) for ${AGENTGYM_ENV_NAME} in env '$TARGET_ENV_NAME'..."
echo "[Server] Base Port: ${AGENTGYM_BASE_PORT}"

# Create logs directory
mkdir -p logs

for (( i=0; i<$NUM_SERVERS; i++ )); do
    # Calculate port for this server instance
    export AGENTGYM_PORT=$((AGENTGYM_BASE_PORT + i))
    AGENTGYM_PORTS+=($AGENTGYM_PORT) # Store port

    # Prepare the specific launch command for this instance
    CURRENT_LAUNCH_CMD=$(eval echo $LAUNCH_CMD) # Substitute $AGENTGYM_PORT

    echo "[Server $(($i+1))/$NUM_SERVERS] Launching on ${AGENTGYM_HOST}:${AGENTGYM_PORT}..."
    echo "[Server $(($i+1))/$NUM_SERVERS] Command: $CURRENT_LAUNCH_CMD"

    # Run server in background using conda run
    LOG_FILE="logs/${TARGET_ENV_NAME}_server_${AGENTGYM_PORT}.log"
    echo "[Server $(($i+1))/$NUM_SERVERS] Logging to $LOG_FILE"

    # Use bash -c to handle potential env vars in launch cmd (like for sqlgym)
    conda run --no-capture-output -n "$TARGET_ENV_NAME" bash -c "$CURRENT_LAUNCH_CMD" > "$LOG_FILE" 2>&1 &
    PID=$!

    # Check if PID was obtained
    if [ -z "$PID" ]; then
        echo "[Error] Failed to get PID for AgentGym server instance $i on port $AGENTGYM_PORT."
        # Attempt to kill already launched servers before exiting
        for p in "${AGENTGYM_PIDS[@]}"; do kill $p 2>/dev/null; done
        exit 1
    fi
    AGENTGYM_PIDS+=($PID) # Store PID
    echo "[Server $(($i+1))/$NUM_SERVERS] Launched (PID: $PID)."
    sleep 2 # Small delay between starting servers
done

# --- Wait and Check Servers ---
echo "[Server] Waiting for AgentGym servers (${AGENTGYM_PIDS[*]}) to initialize..."
sleep 15 # Adjust sleep time if needed

# Check if all server processes are still running
ALL_SERVERS_RUNNING=true
for PID in "${AGENTGYM_PIDS[@]}"; do
    if ! kill -0 $PID > /dev/null 2>&1; then
        echo "[Error] AgentGym server (PID: $PID) failed to start or exited prematurely."
        # Attempt to find the corresponding log file (this is a bit heuristic)
        PORT=$(grep -oP -- "--port\\s+\\K\\d+" "logs/"*"${PID}"* 2>/dev/null || echo "unknown")
        echo "[Error] Check server log potentially named logs/${TARGET_ENV_NAME}_server_${PORT}.log or similar."
        ALL_SERVERS_RUNNING=false
    fi
done

if [ "$ALL_SERVERS_RUNNING" = false ]; then
    echo "[Error] Not all servers started successfully. Exiting."
    # Kill remaining servers
    for p in "${AGENTGYM_PIDS[@]}"; do kill $p 2>/dev/null; done
    exit 1
fi
echo "[Server] All AgentGym servers appear to be running."

# Setup trap to kill all server processes on script exit/interrupt
trap "echo '[Cleanup] Stopping AgentGym servers (PIDs: ${AGENTGYM_PIDS[*]})...'; kill ${AGENTGYM_PIDS[*]} 2>/dev/null || echo '[Cleanup] Servers already stopped.'; wait ${AGENTGYM_PIDS[*]} 2>/dev/null" EXIT

# --- Data and Experiment Naming ---
export DATA_DIR=${DATA_DIR_OVERRIDE:-"data/$AGENTGYM_ENV_NAME"} # Default data dir based on env name
export EXPERIMENT_NAME="OpenManus-rl-ppo-${BASE_MODEL##*/}-${AGENTGYM_ENV_NAME}${EXP_NAME_SUFFIX}"


# --- Run PPO Training in Base Environment ---
echo -e "\\n[Trainer] Running PPO training in base environment '$BASE_CONDA_ENV'..."
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# Construct server base URL, adding path if needed
AGENTGYM_SERVER_BASE="http://$AGENTGYM_HOST" # Base URL without port
# Construct the list of ports as a comma-separated string for OmegaConf
AGENTGYM_PORTS_STR=$(IFS=,; echo "${AGENTGYM_PORTS[*]}")

echo "[Trainer] Using Data Directory: $DATA_DIR"
echo "[Trainer] Experiment Name: $EXPERIMENT_NAME"

echo "[Trainer] AgentGym Base URL: $AGENTGYM_SERVER_BASE"
echo "[Trainer] AgentGym Ports: $AGENTGYM_PORTS_STR" # Pass list of ports

# Check if train/test files exist
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="$DATA_DIR/test.parquet"

if [ ! -f "$TRAIN_FILE" ]; then
    echo "[Warning] Train file not found at $TRAIN_FILE. Ensure data generation script was run for $AGENTGYM_ENV_NAME."
fi
if [ ! -f "$TEST_FILE" ]; then
    echo "[Warning] Test file not found at $TEST_FILE. Ensure data generation script was run for $AGENTGYM_ENV_NAME."
fi

# Ensure base environment is activated correctly for trainer
echo "[Trainer] Ensuring base environment '$BASE_CONDA_ENV' is active..."
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$BASE_CONDA_ENV" || { echo "Error: Failed to activate base env '$BASE_CONDA_ENV'"; exit 1; }

# Check and install dependencies within the base environment
echo "[Trainer] Checking and installing required dependencies in '$BASE_CONDA_ENV'..."
for pkg in tensordict codetiming ray wandb transformers; do
    if ! python -c "import $pkg" &>/dev/null; then
        echo "[Trainer] Installing missing dependency: $pkg"
        pip install $pkg
    fi
done

TRAINER_LOG_FILE="logs/${EXPERIMENT_NAME}.log"
echo "[Trainer] Logging trainer output to $TRAINER_LOG_FILE"
echo "[Trainer] Starting PPO training..."

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \\
    data.train_files=$TRAIN_FILE \\
    data.val_files=$TEST_FILE \\
    data.env_name=$AGENTGYM_ENV_NAME \\
    data.env_server_base=$AGENTGYM_SERVER_BASE \\
    data.env_ports=[${AGENTGYM_PORTS_STR}] \\ // Pass ports as a list
    data.train_data_num=null \\
    data.val_data_num=null \\
    data.train_batch_size=512 \\
    data.val_batch_size=256 \\
    data.max_prompt_length=4096 \\
    data.max_response_length=500 \\
    data.max_start_length=2048 \\
    data.max_obs_length=500 \\
    data.shuffle_train_dataloader=True \\
    algorithm.adv_estimator=gae \\
    actor_rollout_ref.model.path=$BASE_MODEL \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.enable_gradient_checkpointing=true \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \\
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=true \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.n_agent=1 \\
    actor_rollout_ref.rollout.temperature=1 \\
    actor_rollout_ref.actor.state_masking=true \\
    critic.optim.lr=1e-5 \\
    critic.model.use_remove_padding=True \\
    critic.optim.lr_warmup_steps_ratio=0.05 \\
    critic.model.path=$BASE_MODEL \\
    critic.model.enable_gradient_checkpointing=true \\
    critic.ppo_micro_batch_size=8 \\
    critic.model.fsdp_config.param_offload=true \\
    critic.model.fsdp_config.grad_offload=true \\
    critic.model.fsdp_config.optimizer_offload=true \\
    algorithm.kl_ctrl.kl_coef=0.001 \\
    algorithm.no_think_rl=false \\
    algorithm.reward_score_fn=agentgym \\
    trainer.critic_warmup=0 \\
    trainer.logger=['wandb'] \\
    +trainer.val_only=false \\
    +trainer.val_before_train=true \\
    trainer.default_hdfs_dir=null \\
    trainer.n_gpus_per_node=8 \\
    trainer.nnodes=1 \\
    trainer.save_freq=100 \\
    trainer.test_freq=50 \\
    trainer.project_name=$WAND_PROJECT \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.total_epochs=15 \\
    trainer.total_training_steps=305 \\
    trainer.default_hdfs_dir=null \\
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \\
    max_turns=2 \\
    2>&1 | tee "$TRAINER_LOG_FILE" # Log trainer output

TRAINER_EXIT_CODE=$?

echo "PPO training finished with exit code $TRAINER_EXIT_CODE."

# Cleanup is handled by the trap


exit $TRAINER_EXIT_CODE
#!/bin/bash

# --- Configuration (defaults, can be overridden via env vars) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} # adjust GPU IDs here
WAND_PROJECT=${WAND_PROJECT:-'ToT-VERL-Agent'}
export BASE_MODEL=${BASE_MODEL:-'./Qwen2.5-3B'} # 使用当前目录下的相对路径

# 确保使用完全展开的路径 - 将任何~替换为$HOME
if [[ "$BASE_MODEL" == "~"* ]]; then
    BASE_MODEL="${HOME}${BASE_MODEL:1}"
    echo "[Info] Expanded BASE_MODEL path to: $BASE_MODEL"
fi

# 确保路径是绝对路径
if [[ "$BASE_MODEL" != /* && "$BASE_MODEL" != ./* ]]; then
    BASE_MODEL="./$BASE_MODEL"
    echo "[Info] Adjusted BASE_MODEL to relative path: $BASE_MODEL"
fi

AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'} # Default to 0.0.0.0 for external access
AGENTGYM_SQL_BIRD_PATH=${AGENTGYM_SQL_BIRD_PATH:-} # Used only for sqlgym
export PYTHONPATH=".:./openmanus_rl/agentgym/agentenv:${PYTHONPATH}"

# --- Argument Parsing ---
usage() {
    echo "Usage: $0 --env_name <environment_name> [--num_servers <N>] [--base_port <port>] [--data_dir <path>] [--exp_name_suffix <suffix>] [--tot_strategy <BFS/DFS>] [--tot_beam <width>] [--tot_branches <max>] [--use_existing_servers] [--model_path <path>]"
    echo "Supported env_names: webshop, webarena, maze, wordle, alfworld, sciworld, babyai, textcraft, weather, movie, academia, todo, sheet, sqlgym"
    echo "  --num_servers: Number of parallel AgentGym servers to launch (default: 1)."
    echo "  --base_port: Starting port number for servers (default varies by env)."
    echo "  --tot_strategy: Tree of Thought search strategy: BFS or DFS (default: BFS)."
    echo "  --tot_beam: Tree of Thought beam width (default: 3)."
    echo "  --tot_branches: Maximum branches to explore (default: 15)."
    echo "  --use_existing_servers: Skip server launching and use existing servers (provide ports)."
    echo "  --env_conda: Name of the conda environment to use for AgentGym (default: webshop or uses current env)."
    echo "  --model_path: Path to the model directory (default: ./Qwen2.5-3B)."
    exit 1
}

AGENTGYM_ENV_NAME="webshop" # Default environment
NUM_SERVERS=1 # Default number of servers
BASE_PORT_OVERRIDE=""
DATA_DIR_OVERRIDE=""
EXP_NAME_SUFFIX=""
TOT_STRATEGY="BFS"
TOT_BEAM=3
TOT_BRANCHES=15
USE_EXISTING_SERVERS=false
ENV_CONDA_NAME=""
MODEL_PATH_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --env_name)
            AGENTGYM_ENV_NAME="$2"; shift; shift;;
        --num_servers)
            NUM_SERVERS="$2"; shift; shift;;
        --base_port)
            BASE_PORT_OVERRIDE="$2"; shift; shift;;
        --data_dir)
            DATA_DIR_OVERRIDE="$2"; shift; shift;;
        --exp_name_suffix)
            EXP_NAME_SUFFIX="_$2"; shift; shift;;
        --tot_strategy)
            TOT_STRATEGY="$2"; shift; shift;;
        --tot_beam)
            TOT_BEAM="$2"; shift; shift;;
        --tot_branches)
            TOT_BRANCHES="$2"; shift; shift;;
        --use_existing_servers)
            USE_EXISTING_SERVERS=true; shift;;
        --env_conda)
            ENV_CONDA_NAME="$2"; shift; shift;;
        --model_path)
            MODEL_PATH_OVERRIDE="$2"; shift; shift;;
        *)
            echo "Unknown option: $1"; usage;;
    esac
done

# 如果提供了模型路径覆盖，则使用它
if [ -n "$MODEL_PATH_OVERRIDE" ]; then
    # 展开波浪号
    if [[ "$MODEL_PATH_OVERRIDE" == "~"* ]]; then
        MODEL_PATH_OVERRIDE="${HOME}${MODEL_PATH_OVERRIDE:1}"
    fi

    # 确保路径是绝对路径或相对路径
    if [[ "$MODEL_PATH_OVERRIDE" != /* && "$MODEL_PATH_OVERRIDE" != ./* ]]; then
        MODEL_PATH_OVERRIDE="./$MODEL_PATH_OVERRIDE"
    fi

    BASE_MODEL="$MODEL_PATH_OVERRIDE"
    echo "[Info] Using model path from command line: $BASE_MODEL"
fi

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
echo "[Info] VERL trainer will run in this environment."

# --- Environment Specific Setup ---
LAUNCH_CMD=""
DEFAULT_BASE_PORT=""
URL_PATH=""

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

# Determine base port
AGENTGYM_BASE_PORT=${BASE_PORT_OVERRIDE:-$DEFAULT_BASE_PORT}
AGENTGYM_PORTS=() # Array to store ports
AGENTGYM_PGIDS=() # Array to store PGIDs (only used if launching new servers)

if [ "$USE_EXISTING_SERVERS" = true ]; then
    # Use existing servers - just create port list
    for (( i=0; i<$NUM_SERVERS; i++ )); do
        export AGENTGYM_PORT=$((AGENTGYM_BASE_PORT + i))
        AGENTGYM_PORTS+=($AGENTGYM_PORT)
    done
    echo "[Info] Using existing servers on ports: ${AGENTGYM_PORTS[*]}"
else
    # --- Start AgentGym Servers in Dedicated Environment ---
    # Determine target environment name
    if [ -z "$ENV_CONDA_NAME" ]; then
        # If no specific env provided, try the default naming scheme first
        TARGET_ENV_NAME="agentenv-${AGENTGYM_ENV_NAME}"
        # Check if it exists, otherwise fall back to environment specific default
        if ! conda env list | grep -Eq "^${TARGET_ENV_NAME}\\s"; then
            case $AGENTGYM_ENV_NAME in
                webshop)
                    TARGET_ENV_NAME="webshop";;
                *)
                    TARGET_ENV_NAME=$BASE_CONDA_ENV;;
            esac
        fi
    else
        # Use the provided environment name
        TARGET_ENV_NAME=$ENV_CONDA_NAME
    fi

    # Check if target env exists
    if ! conda env list | grep -Eq "^${TARGET_ENV_NAME}\\s"; then
        echo "[Error] Environment '$TARGET_ENV_NAME' not found. Please specify a valid conda environment with --env_conda."
        exit 1
    fi

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

        # Run server in background using conda run within a new process group
        LOG_FILE="logs/${TARGET_ENV_NAME}_server_${AGENTGYM_PORT}.log"
        echo "[Server $(($i+1))/$NUM_SERVERS] Logging to $LOG_FILE"

        # Use setsid to ensure the server runs in its own process group
        setsid conda run --no-capture-output -n "$TARGET_ENV_NAME" bash -c "$CURRENT_LAUNCH_CMD" > "$LOG_FILE" 2>&1 &
        PGID=$! # PID of setsid becomes the Process Group ID

        # Check if PGID was obtained
        if [ -z "$PGID" ]; then
            echo "[Error] Failed to get PGID for AgentGym server instance $i on port $AGENTGYM_PORT."
            # Attempt to kill already launched servers before exiting
            for pgid in "${AGENTGYM_PGIDS[@]}"; do kill -- -$pgid 2>/dev/null; done # Kill process group
            exit 1
        fi
        AGENTGYM_PGIDS+=($PGID) # Store PGID
        echo "[Server $(($i+1))/$NUM_SERVERS] Launched (PGID: $PGID)."
        sleep 2 # Small delay between starting servers
    done

    # --- Wait and Check Servers ---
    echo "[Server] Checking if AgentGym servers (${AGENTGYM_PORTS[*]}) are responsive..."
    ALL_SERVERS_RUNNING=true
    MAX_RETRIES=5
    RETRY_DELAY=3
    CONNECT_TIMEOUT=1

    for (( i=0; i<${#AGENTGYM_PORTS[@]}; i++ )); do
        PORT=${AGENTGYM_PORTS[i]}
        PGID=${AGENTGYM_PGIDS[i]} 
        LOG_FILE="logs/${TARGET_ENV_NAME}_server_${PORT}.log"
        SERVER_UP=false

        # Determine host to check (use localhost if host is 0.0.0.0)
        CHECK_HOST=$AGENTGYM_HOST
        if [ "$CHECK_HOST" == "0.0.0.0" ]; then
            CHECK_HOST="127.0.0.1"
        fi

        echo "[Server Check] Checking server on ${CHECK_HOST}:${PORT} (PGID: $PGID)..."
        for (( attempt=1; attempt<=$MAX_RETRIES; attempt++ )); do
            # Use netcat to check if port is open
            if nc -z -w $CONNECT_TIMEOUT "$CHECK_HOST" "$PORT" > /dev/null 2>&1; then
                 echo "[Server Check] Server on port $PORT is responsive."
                 SERVER_UP=true
                 break # Exit retry loop for this server
            else
                if [ $attempt -lt $MAX_RETRIES ]; then
                    echo "[Server Check] Server on port $PORT not responsive (Attempt $attempt/$MAX_RETRIES). Retrying in $RETRY_DELAY seconds..."
                    sleep $RETRY_DELAY
                else
                    echo "[Error] Server on port $PORT (PGID: $PGID) failed to respond after $MAX_RETRIES attempts."
                    echo "[Error] Check server log for details: $LOG_FILE"
                fi
            fi
        done

        if [ "$SERVER_UP" = false ]; then
            ALL_SERVERS_RUNNING=false
            break
        fi
    done

    if [ "$ALL_SERVERS_RUNNING" = false ]; then
        echo "[Error] Not all AgentGym servers started successfully. Initiating cleanup..."
        # Cleanup process groups
        CLEANUP_PGIDS_ON_FAIL=(${AGENTGYM_PGIDS[*]});
        for pgid_fail in "${CLEANUP_PGIDS_ON_FAIL[@]}"; do
            echo "[Cleanup] Killing process group -$pgid_fail due to failed startup check."
            kill -- -$pgid_fail 2>/dev/null;
        done
        wait 2>/dev/null # Give kill commands a moment
        echo "[Error] Exiting script due to server startup failure."
        exit 1 # Exit with error code
    fi

    echo "[Server] All AgentGym servers appear to be responsive and running."

    # Setup trap to kill all server process groups on script exit/interrupt
    trap "echo '[Cleanup] Stopping AgentGym server process groups (PGIDs: ${AGENTGYM_PGIDS[*]})...'; CLEANUP_PGIDS=(${AGENTGYM_PGIDS[*]}); for pgid in \${CLEANUP_PGIDS[@]}; do echo '[Cleanup] Killing process group -\$pgid'; kill -- -\$pgid 2>/dev/null; done; wait 2>/dev/null; echo '[Cleanup] Done.'" EXIT
fi

# --- Data and Experiment Naming ---
export DATA_DIR=${DATA_DIR_OVERRIDE:-"./data/$AGENTGYM_ENV_NAME"} # Default data dir based on env name
export EXPERIMENT_NAME="ToT-Agent-${BASE_MODEL##*/}-${AGENTGYM_ENV_NAME}${EXP_NAME_SUFFIX}"

# --- Run ToT Training in Base Environment ---
echo -e "\\n[Trainer] Running ToT training in base environment '$BASE_CONDA_ENV'..."
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

# Construct server base URL, adding path if needed
AGENTGYM_SERVER_BASE="http://$AGENTGYM_HOST" # Base URL without port
# Construct the list of ports as a comma-separated string for OmegaConf
AGENTGYM_PORTS_STR=$(IFS=,; echo "${AGENTGYM_PORTS[*]}")

echo "[Trainer] Using Model Path: $BASE_MODEL"
echo "[Trainer] Using Data Directory: $DATA_DIR"
echo "[Trainer] Experiment Name: $EXPERIMENT_NAME"
echo "[Trainer] AgentGym Base URL: $AGENTGYM_SERVER_BASE"
echo "[Trainer] AgentGym Ports: $AGENTGYM_PORTS_STR"
echo "[Trainer] ToT Strategy: $TOT_STRATEGY"
echo "[Trainer] ToT Beam Width: $TOT_BEAM"
echo "[Trainer] ToT Max Branches: $TOT_BRANCHES"

# Check if train/test files exist
TRAIN_FILE="/hdd/yxyang/OpenManus-RL-main/data/webshop/train.parquet"
TEST_FILE="/hdd/yxyang/OpenManus-RL-main/data/webshop/val.parquet"

echo "[Trainer] Train file: $TRAIN_FILE"
echo "[Trainer] Test file: $TEST_FILE"

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
echo "[Trainer] Starting ToT training..."

# --- Construct Hydra Overrides Array for ToT ---
hydra_overrides=(
    "data.train_files=$TRAIN_FILE"
    "data.val_files=$TEST_FILE"
    "data.env_name=$AGENTGYM_ENV_NAME"
    "data.env_server_base=$AGENTGYM_SERVER_BASE"
    "data.env_ports=[${AGENTGYM_PORTS_STR}]"
    "data.train_data_num=null"
    "data.val_data_num=null"
    "data.train_batch_size=8"  # Smaller batch size for ToT
    "data.val_batch_size=4"
    "data.max_prompt_length=1024"
    "data.max_response_length=512"
    "data.max_start_length=512"
    "data.max_obs_length=512"
    "data.shuffle_train_dataloader=True"
    "tot.beam_width=$TOT_BEAM"
    "tot.exploration_factor=2"
    "tot.max_branches=$TOT_BRANCHES"
    "tot.value_guidance=True"
    "tot.temperature=1.0"
    "tot.search_strategy=$TOT_STRATEGY"
    "tot.value_threshold=0.0"
    "tot.reward_cutoff=0.8"
    "algorithm.adv_estimator=gae"
    "algorithm.reward_score_fn=agentgym" 
    "actor_rollout_ref.model.path=$BASE_MODEL"
    "actor_rollout_ref.actor.optim.lr=5e-6"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.model.use_remove_padding=True"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05"
    "actor_rollout_ref.actor.state_masking=true"
    "actor_rollout_ref.actor.ppo_mini_batch_size=4"
    "actor_rollout_ref.actor.ppo_micro_batch_size=2"
    "actor_rollout_ref.actor.fsdp_config.param_offload=true"
    "actor_rollout_ref.actor.fsdp_config.grad_offload=true"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
    "+actor_rollout_ref.model.torch_dtype=bfloat16"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=64"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7"
    "actor_rollout_ref.ref.log_prob_micro_batch_size=64"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "actor_rollout_ref.rollout.n_agent=1" 
    "actor_rollout_ref.rollout.temperature=1"
    "critic.optim.lr=5e-6"
    "critic.model.use_remove_padding=True"
    "critic.optim.lr_warmup_steps_ratio=0.05"
    "critic.model.path=$BASE_MODEL"
    "critic.model.enable_gradient_checkpointing=true"
    "critic.ppo_micro_batch_size=2"
    "critic.model.fsdp_config.param_offload=true"
    "critic.model.fsdp_config.grad_offload=true"
    "critic.model.fsdp_config.optimizer_offload=true"
    "+critic.model.torch_dtype=bfloat16"
    "algorithm.kl_ctrl.kl_coef=0.001"
    "algorithm.no_think_rl=false"
    "trainer.critic_warmup=0"
    "trainer.logger=['wandb', 'console']"
    "+trainer.val_only=false"
    "+trainer.val_before_train=true"
    "trainer.default_hdfs_dir=null"
    "trainer.n_gpus_per_node=${#CUDA_VISIBLE_DEVICES//,/ }"  # Count number of GPUs
    "trainer.nnodes=1"
    "trainer.save_freq=5"
    "trainer.test_freq=5"
    "trainer.project_name=$WAND_PROJECT"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.total_epochs=20"
    "trainer.total_training_steps=null"
    "trainer.default_hdfs_dir=null"
    "trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME"
    "max_turns=10"
)

# --- Execute Python Training Script ---
# 修改为运行train_tot_agent.py脚本 (不是main.train_tot_agent)
PYTHONUNBUFFERED=1 python3 train_tot_agent.py \
    --config-name tot_ppo_trainer --config-path openmanus_tot \
    "${hydra_overrides[@]}" \
    2>&1 | tee "$TRAINER_LOG_FILE" # Log trainer output

TRAINER_EXIT_CODE=$?

echo "ToT training finished with exit code $TRAINER_EXIT_CODE."

# Cleanup is handled by the trap

exit $TRAINER_EXIT_CODE
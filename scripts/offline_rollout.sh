CONFIG_FILE="" # fulfill the config yaml file here
MODEL_PATH=""
OUTPUT_DIR=""
TASK_NAMES=""
DATA_LEN=200
TIMEOUT=2400
DO_SAMPLE="False"
TEMPERATURE=1.0
SEED=42
DEBUG=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --task_names)
      TASK_NAMES="$2"
      shift 2
      ;;
    --data_len)
      DATA_LEN="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --do_sample)
      DO_SAMPLE="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build command
CMD="python traj_generation/rollout_eval.py --config $CONFIG_FILE"

if [ ! -z "$MODEL_PATH" ]; then
  CMD="$CMD --model_path $MODEL_PATH"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
  CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ ! -z "$TASK_NAMES" ]; then
  CMD="$CMD --task_names $TASK_NAMES"
fi

CMD="$CMD --data_len $DATA_LEN --timeout $TIMEOUT --do_sample $DO_SAMPLE --temperature $TEMPERATURE --seed $SEED"

if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug"
fi

# Create log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="./logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/offline_rollout_$TIMESTAMP.log"

# Print the command
echo "Running: $CMD"
echo "Logging to: $LOG_FILE"

# Execute with logging
eval "$CMD | tee $LOG_FILE"

echo "Evaluation complete! Results saved to the output directory."
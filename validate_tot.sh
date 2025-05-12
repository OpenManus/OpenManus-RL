#!/bin/bash

# 验证 Tree of Thought Agent 实现的运行脚本

# --- Configuration (defaults, can be overridden via env vars) ---
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} # adjust GPU IDs here
export BASE_MODEL=${BASE_MODEL:-'./Qwen2.5-3B'} # Path to your base model
AGENTGYM_HOST=${AGENTGYM_HOST:-'0.0.0.0'} # Default to 0.0.0.0 for external access
export PYTHONPATH=".:./openmanus_rl/agentgym/agentenv:${PYTHONPATH}"

# --- Argument Parsing ---
usage() {
    echo "Usage: $0 [--model_path <path>] [--env_name <env>] [--env_port <port>] [--tot_strategy <BFS|DFS>] [--tot_beam <width>] [--tot_branches <max>] [--max_turns <turns>] [--num_examples <n>] [--debug]"
    echo "  --model_path: Path to the model (default: ./Qwen2.5-3B)"
    echo "  --env_name: Environment name (default: webshop)"
    echo "  --env_port: Environment server port (default: 36001)"
    echo "  --tot_strategy: ToT search strategy: BFS or DFS (default: BFS)"
    echo "  --tot_beam: ToT beam width (default: 3)"
    echo "  --tot_branches: Maximum branches to explore (default: 10)"
    echo "  --max_turns: Maximum number of turns (default: 5)"
    echo "  --num_examples: Number of examples to test (default: 2)"
    echo "  --debug: Enable additional debug output"
    exit 1
}

MODEL_PATH=${BASE_MODEL}
ENV_NAME="webshop"
ENV_PORT=36001
TOT_STRATEGY="BFS"
TOT_BEAM=3
TOT_BRANCHES=10
MAX_TURNS=5
NUM_EXAMPLES=2
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
            MODEL_PATH="$2"; shift; shift;;
        --env_name)
            ENV_NAME="$2"; shift; shift;;
        --env_port)
            ENV_PORT="$2"; shift; shift;;
        --tot_strategy)
            TOT_STRATEGY="$2"; shift; shift;;
        --tot_beam)
            TOT_BEAM="$2"; shift; shift;;
        --tot_branches)
            TOT_BRANCHES="$2"; shift; shift;;
        --max_turns)
            MAX_TURNS="$2"; shift; shift;;
        --num_examples)
            NUM_EXAMPLES="$2"; shift; shift;;
        --debug)
            DEBUG_FLAG="--debug"; shift;;
        -h|--help)
            usage;;
        *)
            echo "Unknown option: $1"; usage;;
    esac
done

# 确保波浪号展开
if [[ "$MODEL_PATH" == "~"* ]]; then
    MODEL_PATH="${HOME}${MODEL_PATH:1}"
    echo "[Info] Expanded model path to: $MODEL_PATH"
fi

# 确保路径格式正确
if [[ "$MODEL_PATH" != /* && "$MODEL_PATH" != ./* ]]; then
    MODEL_PATH="./$MODEL_PATH"
    echo "[Info] Adjusted model path to: $MODEL_PATH"
fi

# 创建日志目录
mkdir -p logs

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/tot_validation_${TIMESTAMP}.log"

echo "==============================================================="
echo "📋 Tree of Thought Agent 验证"
echo "==============================================================="
echo "🔍 参数:"
echo "  📂 模型路径: $MODEL_PATH"
echo "  🌍 环境: $ENV_NAME"
echo "  🔌 环境端口: $ENV_PORT"
echo "  🔀 ToT 策略: $TOT_STRATEGY"
echo "  🔢 ToT 束宽: $TOT_BEAM"
echo "  🌲 最大分支数: $TOT_BRANCHES"
echo "  🔄 最大回合数: $MAX_TURNS"
echo "  🧪 测试示例数: $NUM_EXAMPLES"
echo "  📝 日志文件: $LOG_FILE"
echo "==============================================================="

# 检查服务器是否运行
CHECK_HOST="127.0.0.1"
if nc -z -w 1 "$CHECK_HOST" "$ENV_PORT" > /dev/null 2>&1; then
    echo "✅ 检测到环境服务器正在运行在端口 $ENV_PORT"
else
    echo "⚠️ 警告: 未检测到环境服务器在端口 $ENV_PORT 上运行"
    echo "   您可能需要先启动环境服务器。例如:"
    echo "   webshop --port $ENV_PORT"
    
    # 询问是否仍要继续
    read -p "是否仍要继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "终止验证。请先启动环境服务器。"
        exit 1
    fi
fi

# 确保正确的conda环境
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
CURRENT_ENV=${CONDA_DEFAULT_ENV:-"base"}
echo "📦 当前conda环境: $CURRENT_ENV"

# 运行验证脚本
echo "🚀 开始验证... 输出将保存到 $LOG_FILE"
PYTHONUNBUFFERED=1 python validate_tot_agent.py \
    --model_path "$MODEL_PATH" \
    --env_name "$ENV_NAME" \
    --env_port "$ENV_PORT" \
    --tot_strategy "$TOT_STRATEGY" \
    --tot_beam "$TOT_BEAM" \
    --tot_branches "$TOT_BRANCHES" \
    --max_turns "$MAX_TURNS" \
    --num_examples "$NUM_EXAMPLES" \
    $DEBUG_FLAG \
    2>&1 | tee "$LOG_FILE"

VALIDATION_EXIT_CODE=${PIPESTATUS[0]}

if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    echo "✅ 验证完成!"
    echo "📋 结果保存在 tot_validation_results.json"
else
    echo "❌ 验证失败，退出代码: $VALIDATION_EXIT_CODE"
    echo "📋 错误日志在 $LOG_FILE"
fi

exit $VALIDATION_EXIT_CODE
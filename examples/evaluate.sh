#!/bin/bash

# Agent Trajectory Evaluation Script
# Usage: ./evaluate.sh [options]

set -e

# Default configuration
INPUT_DIR="../experiments/chat_histories"
OUTPUT_DIR="../evaluation_results1"
MAX_CONCURRENT=5
MODEL="gpt-4o"
TEMPERATURE=0.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -k|--api-key)
            export OPENAI_API_KEY="$2"
            shift 2
            ;;
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Agent Trajectory Evaluation Script"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -i, --input-dir DIR     Input directory containing trajectory JSON files"
            echo "                          (default: ../trajectories/chat_histories)"
            echo "  -o, --output-dir DIR    Output directory for evaluation results"
            echo "                          (default: ../evaluation_results)"
            echo "  -c, --max-concurrent N  Maximum concurrent evaluations (default: 5)"
            echo "  -m, --model MODEL       LLM model to use (default: gpt-4o)"
            echo "  -t, --temperature T     Temperature for LLM (default: 0.0)"
            echo "  -k, --api-key KEY       OpenAI API key (or set OPENAI_API_KEY env var)"
            echo "      --api-url URL       Custom API URL"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -i ./trajectories -o ./results -c 3"
            echo "  $0 -m gpt-4-turbo -t 0.1 -k your-api-key"
            echo "  $0 --input-dir /path/to/trajs --max-concurrent 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OpenAI API key not set"
    echo "Please set OPENAI_API_KEY environment variable or use -k option"
    echo "Example: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

# Count trajectory files
TRAJ_COUNT=$(find "$INPUT_DIR" -name "*.json" | wc -l)
if [ "$TRAJ_COUNT" -eq 0 ]; then
    echo "Error: No JSON files found in '$INPUT_DIR'"
    exit 1
fi

# Display configuration
echo "=================================================="
echo "Agent Trajectory Evaluation"
echo "=================================================="
echo "Input Directory:     $INPUT_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Trajectory Files:    $TRAJ_COUNT"
echo "Model:               $MODEL"
echo "Temperature:         $TEMPERATURE"
echo "Max Concurrent:      $MAX_CONCURRENT"
echo "API Key:             ${OPENAI_API_KEY:0:8}..." # Show only first 8 chars
echo "=================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python trajectory_evaluator.py"
CMD="$CMD --input-dir '$INPUT_DIR'"
CMD="$CMD --output-dir '$OUTPUT_DIR'"
CMD="$CMD --model '$MODEL'"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --max-concurrent $MAX_CONCURRENT"

if [ ! -z "$API_URL" ]; then
    CMD="$CMD --api-url '$API_URL'"
fi

# Run evaluation
echo "Starting evaluation..."
echo "Command: $CMD"
echo ""

eval $CMD

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Evaluation completed successfully!"
    echo "=================================================="
    echo "Results location: $OUTPUT_DIR"
    echo ""
    
    # Show quick summary if summary file exists
    if [ -f "$OUTPUT_DIR/evaluation_summary.json" ]; then
        echo "Quick Summary:"
        python3 -c "
import json
with open('$OUTPUT_DIR/evaluation_summary.json') as f:
    data = json.load(f)
overview = data['overview']
print(f\"  Total Trajectories: {overview['total_trajectories']}\")
print(f\"  Success Rate: {overview['success_rate']:.1%}\")
print(f\"  Average Score: {overview['average_overall_score']:.2f}/5.0\")
print()
print('Module Performance:')
for module, stats in data['module_performance'].items():
    print(f\"  {module.replace('_', ' ').title()}: {stats['average_score']:.2f}/5.0\")
"
    fi
    
    echo ""
    echo "Next steps:"
    echo "  1. Review individual evaluations in $OUTPUT_DIR/"
    echo "  2. Check evaluation_summary.json for overall insights"
    echo "  3. Use results to improve agent training"
    
else
    echo ""
    echo "=================================================="
    echo "Evaluation failed!"
    echo "=================================================="
    echo "Please check the error messages above and:"
    echo "  1. Verify API key is correct"
    echo "  2. Check input directory contains valid JSON files"
    echo "  3. Ensure network connectivity"
    echo "  4. Review any error logs"
fi
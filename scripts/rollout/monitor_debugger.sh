#!/bin/bash

# Monitor script for OpenManus Rollout Debugger
# This script monitors the progress of debugger experiments

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}    OpenManus Debugger Monitor         ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Function to show running debugger processes
show_running_debuggers() {
    echo -e "${GREEN}Running Debugger Processes:${NC}"
    ps aux | grep 'openmanus_rollout_debugger' | grep -v grep | while read line; do
        if [ ! -z "$line" ]; then
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')

            # Extract model and debugger model
            model=$(echo "$line" | grep -o "\-\-model [^ ]*" | sed "s/--model //g")
            debugger_model=$(echo "$line" | grep -o "\-\-debugger_model [^ ]*" | sed "s/--debugger_model //g")
            experiment_dir=$(echo "$line" | grep -o "\-\-experiment_dir [^ ]*" | sed "s/--experiment_dir //g")

            printf "  PID: %-8s CPU: %5s%% MEM: %5s%%\n" "$pid" "$cpu" "$mem"
            [ ! -z "$model" ] && printf "    Model: %s\n" "$model"
            [ ! -z "$debugger_model" ] && printf "    Debugger: %s\n" "$debugger_model"
            [ ! -z "$experiment_dir" ] && printf "    Experiment: %s\n" "$experiment_dir"
        fi
    done

    # Check if any debugger is running
    if ! ps aux | grep -q '[o]penmanus_rollout_debugger'; then
        echo "  No debugger processes running"
    fi
    echo ""
}

# Function to find latest experiment directory
find_latest_experiment() {
    # Look for experiment directories matching the pattern
    # When running from scripts/rollout, need to go up two levels
    latest_exp=$(ls -dt ../../experiments/unified_debug_* 2>/dev/null | head -1)

    if [ -z "$latest_exp" ]; then
        # Try to find from running process
        experiment_dir=$(ps aux | grep 'openmanus_rollout_debugger' | grep -v grep | grep -o "\-\-experiment_dir [^ ]*" | sed "s/--experiment_dir //g" | head -1)
        if [ ! -z "$experiment_dir" ] && [ -d "$experiment_dir" ]; then
            latest_exp="$experiment_dir"
        fi
    fi

    echo "$latest_exp"
}

# Function to analyze experiment progress
show_experiment_progress() {
    local exp_dir="$1"

    if [ -z "$exp_dir" ] || [ ! -d "$exp_dir" ]; then
        echo -e "${YELLOW}No experiment directory found${NC}"
        return
    fi

    echo -e "${GREEN}Experiment Progress:${NC}"
    echo "  Directory: $exp_dir"

    # Check for trajectories directory
    traj_dir="$exp_dir/trajectories"
    if [ -d "$traj_dir" ]; then
        total_tasks=$(ls -d "$traj_dir"/*/ 2>/dev/null | wc -l)
        echo "  Total tasks processed: $total_tasks"

        # Count successful tasks
        successful=0
        failed=0
        with_retries=0
        total_retries=0
        debugger_calls=0

        for task_dir in "$traj_dir"/*/; do
            [ -d "$task_dir" ] || continue

            # Check task summary
            if [ -f "$task_dir/task_summary.json" ]; then
                # Check if task was successful
                if grep -q '"won": true' "$task_dir/task_summary.json" 2>/dev/null; then
                    successful=$((successful + 1))
                else
                    failed=$((failed + 1))
                fi

                # Count retries
                retry_count=$(grep -o '"retry_count": [0-9]*' "$task_dir/task_summary.json" 2>/dev/null | grep -o '[0-9]*' | head -1)
                if [ ! -z "$retry_count" ] && [ "$retry_count" -gt 1 ]; then
                    with_retries=$((with_retries + 1))
                    total_retries=$((total_retries + retry_count - 1))
                fi
            fi

            # Count debugger analysis calls
            if [ -f "$task_dir/debugger_analysis_calls.jsonl" ]; then
                local_calls=$(wc -l < "$task_dir/debugger_analysis_calls.jsonl" 2>/dev/null || echo 0)
                debugger_calls=$((debugger_calls + local_calls))
            fi
        done

        # Calculate success rate
        if [ "$total_tasks" -gt 0 ]; then
            success_rate=$(awk -v s="$successful" -v t="$total_tasks" 'BEGIN{printf("%.1f", s*100/t)}')
            printf "  ${CYAN}Success rate: %d/%d (%.1f%%)${NC}\n" "$successful" "$total_tasks" "$success_rate"
        fi

        [ "$failed" -gt 0 ] && echo "  Failed tasks: $failed"
        [ "$with_retries" -gt 0 ] && echo "  Tasks with retries: $with_retries (Total retries: $total_retries)"
        [ "$debugger_calls" -gt 0 ] && echo "  Debugger analysis calls: $debugger_calls"
    fi
    echo ""
}

# Function to show debugger error analysis
show_debugger_analysis() {
    local exp_dir="$1"

    if [ -z "$exp_dir" ] || [ ! -d "$exp_dir/trajectories" ]; then
        return
    fi

    echo -e "${GREEN}Debugger Error Analysis:${NC}"

    # Collect error types from debug analysis files
    declare -A error_types
    declare -A error_modules

    for debug_file in "$exp_dir/trajectories"/*/debug_analysis_retry_*.json; do
        [ -f "$debug_file" ] || continue

        # Extract error type and module
        error_type=$(grep -o '"failure_type": "[^"]*"' "$debug_file" 2>/dev/null | cut -d'"' -f4 | head -1)
        error_module=$(grep -o '"critical_module": "[^"]*"' "$debug_file" 2>/dev/null | cut -d'"' -f4 | head -1)

        [ ! -z "$error_type" ] && ((error_types["$error_type"]++))
        [ ! -z "$error_module" ] && ((error_modules["$error_module"]++))
    done

    # Display error type distribution
    if [ ${#error_types[@]} -gt 0 ]; then
        echo "  Error Types Detected:"
        for error in "${!error_types[@]}"; do
            printf "    %-30s: %d\n" "$error" "${error_types[$error]}"
        done
        echo ""
    fi

    # Display error module distribution
    if [ ${#error_modules[@]} -gt 0 ]; then
        echo "  Critical Modules:"
        for module in "${!error_modules[@]}"; do
            printf "    %-20s: %d\n" "$module" "${error_modules[$module]}"
        done
    fi
    echo ""
}

# Function to show recent log activity
show_recent_logs() {
    echo -e "${GREEN}Recent Log Activity:${NC}"

    # Find the most recent log file
    # When running from scripts/rollout, need to go up two levels
    latest_log=$(ls -t ../../logs/*/unified_run_*.log 2>/dev/null | head -1)

    if [ -f "$latest_log" ]; then
        echo "  Log file: $latest_log"
        echo ""

        # Show last few relevant lines (filter out too verbose debug messages)
        echo "  Recent activity:"
        tail -20 "$latest_log" 2>/dev/null | grep -E "(Env |SUCCESS|FAILED|Attempt |Debugger analysis|Root cause|Task )" | tail -10 | sed 's/^/    /'
    else
        echo "  No log files found"
    fi
    echo ""
}

# Function to show task-specific details
show_task_details() {
    local exp_dir="$1"
    local max_show=5

    if [ -z "$exp_dir" ] || [ ! -d "$exp_dir/trajectories" ]; then
        return
    fi

    echo -e "${GREEN}Recent Task Details (last $max_show):${NC}"

    # Get most recent tasks
    recent_tasks=$(ls -dt "$exp_dir/trajectories"/*/ 2>/dev/null | head -$max_show)

    for task_dir in $recent_tasks; do
        [ -d "$task_dir" ] || continue

        task_name=$(basename "$task_dir")
        echo "  $task_name:"

        # Check attempts
        attempt_count=$(ls "$task_dir"/attempt_*_trajectory.json 2>/dev/null | wc -l)
        if [ "$attempt_count" -eq 0 ]; then
            attempt_count=$(ls "$task_dir"/debug_analysis_retry_*.json 2>/dev/null | wc -l)
            [ "$attempt_count" -gt 0 ] && attempt_count=$((attempt_count + 1))
        fi
        [ "$attempt_count" -eq 0 ] && attempt_count=1

        # Check success
        success="FAILED"
        if [ -f "$task_dir/task_summary.json" ] && grep -q '"won": true' "$task_dir/task_summary.json" 2>/dev/null; then
            success="SUCCESS"
        fi

        # Get last error if exists
        last_debug=$(ls -t "$task_dir"/debug_analysis_retry_*.json 2>/dev/null | head -1)
        if [ -f "$last_debug" ]; then
            error_type=$(grep -o '"failure_type": "[^"]*"' "$last_debug" | cut -d'"' -f4 | head -1)
            printf "    Attempts: %d | Status: %-7s | Last error: %s\n" "$attempt_count" "$success" "$error_type"
        else
            printf "    Attempts: %d | Status: %-7s\n" "$attempt_count" "$success"
        fi
    done
    echo ""
}

# Function to show real-time stats
show_realtime_stats() {
    local exp_dir="$1"

    if [ -z "$exp_dir" ] || [ ! -d "$exp_dir" ]; then
        return
    fi

    echo -e "${GREEN}Real-time Statistics:${NC}"

    # Count files to estimate progress
    trajectory_files=$(find "$exp_dir/trajectories" -name "*.json" 2>/dev/null | wc -l)
    analysis_files=$(find "$exp_dir/trajectories" -name "debugger_*.jsonl" 2>/dev/null | wc -l)

    echo "  Total trajectory files: $trajectory_files"
    echo "  Debugger log files: $analysis_files"

    # Estimate rate
    if [ -f "$exp_dir/.start_time" ]; then
        start_time=$(cat "$exp_dir/.start_time")
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ "$elapsed" -gt 0 ] && [ "$trajectory_files" -gt 0 ]; then
            rate=$(awk -v t="$trajectory_files" -v e="$elapsed" 'BEGIN{printf("%.2f", t*60/e)}')
            echo "  Processing rate: $rate tasks/minute"
        fi
    fi
    echo ""
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}    OpenManus Debugger Monitor         ${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"

    show_running_debuggers

    # Find latest experiment
    exp_dir=$(find_latest_experiment)
    if [ ! -z "$exp_dir" ]; then
        show_experiment_progress "$exp_dir"
        show_debugger_analysis "$exp_dir"
        show_task_details "$exp_dir"
        show_realtime_stats "$exp_dir"
    fi

    show_recent_logs

    echo -e "${BLUE}----------------------------------------${NC}"
    echo "Press Ctrl+C to exit, refreshing in 5 seconds..."
    sleep 5
done
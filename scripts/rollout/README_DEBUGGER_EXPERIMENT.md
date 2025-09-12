# Debugger Experiment System

## Overview
This system runs experiments with an LLM-based debugger that analyzes failed trajectories and provides feedback for retry attempts. It compares first-attempt success rates with debugger-assisted success rates.

## Key Features
- **Multi-attempt execution**: Runs tasks with up to N retry attempts
- **LLM debugger analysis**: Analyzes failed trajectories to identify errors
- **Targeted feedback injection**: Injects feedback at critical failure points
- **Comprehensive tracking**: Tracks both first-attempt and debugger-assisted success rates
- **Organized output structure**: Each task gets its own directory with all attempts

## Directory Structure
```
experiments/
└── alfworld_debug_YYYYMMDD_HHMMSS/
    ├── trajectories/
    │   ├── alfworld_b001_e001_pick_and_place_task/
    │   │   ├── attempt_1_trajectory.json
    │   │   ├── attempt_2_trajectory.json
    │   │   ├── debug_analysis_retry_2.json
    │   │   └── task_summary.json
    │   └── ... (one folder per task)
    └── summaries/
        ├── all_trajectories.jsonl
        └── experiment_summary.json
```

## Usage

### Basic Run (100 Tasks with Debugger)
```bash
./scripts/rollout/test_unified_debugger.sh
```

This will:
- Run 100 AlfWorld tasks
- Use temperature 0.0 for deterministic outputs
- Enable debugger with up to 3 retry attempts
- Save all trajectories and debug analyses

### Custom Configuration
```bash
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --batch_size 10 \
    --total_envs 100 \
    --max_steps 50 \
    --model gpt-4o-mini \
    --temperature 0.0 \
    --enable_debugger \
    --max_debug_retry 3 \
    --debugger_model gpt-4o \
    --debugger_temperature 0.0 \
    --experiment_dir experiments/my_experiment \
    --save_all_attempts \
    --save_per_task_trajectories
```

## Key Parameters

### Execution
- `--total_envs`: Number of tasks to run (default: 100)
- `--batch_size`: Number of parallel environments per batch
- `--max_steps`: Maximum steps per episode
- `--temperature`: LLM temperature (0.0 for deterministic)

### Debugger
- `--enable_debugger`: Enable the LLM debugger
- `--max_debug_retry`: Maximum retry attempts (default: 5)
- `--debugger_model`: Model for debugging (default: gpt-4o)
- `--debugger_temperature`: Temperature for debugger

### Output
- `--experiment_dir`: Root directory for all outputs
- `--save_per_task_trajectories`: Create separate folder per task
- `--save_all_attempts`: Save all retry trajectories

## Output Files

### Task-level Files
Each task directory contains:
- `attempt_N_trajectory.json`: Full trajectory for attempt N
- `debug_analysis_retry_N.json`: Debugger analysis for retry N
- `task_summary.json`: Summary of all attempts with success status

### Experiment-level Files
- `summaries/all_trajectories.jsonl`: All trajectories in JSONL format
- `summaries/experiment_summary.json`: Overall experiment statistics

## Metrics Reported

### Success Rates
1. **First Attempt Success Rate**: Success rate on the first try
2. **Debugger-Assisted Success Rate**: Success rate after retries with debugger
3. **Improvement**: Percentage improvement from debugger assistance

### Example Output
```
========== Success Rate Analysis ==========
First Attempt Success Rate: 0.3200 (32/100)
Success Rate with Debugger: 0.5400 (54/100)
Improvement from Debugger: +0.2200 (22.00%)
```

## Supported Environments
- **AlfWorld**: Text-based household tasks
- **GAIA**: Tool-use reasoning tasks
- **WebShop**: Web shopping navigation

## Notes
- Temperature is set to 0.0 for reproducible results
- Each task gets a unique identifier based on its content
- Debug analyses show exact failure points and suggestions
- All timestamps use YYYYMMDD_HHMMSS format

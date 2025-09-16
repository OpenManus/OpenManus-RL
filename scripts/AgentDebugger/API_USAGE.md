# Agent Error Detector API - Integration Guide

## Quick Start

```python
from api_interface import analyze_trajectory_sync
import json

# Load your trajectory
with open('trajectory.json', 'r') as f:
    trajectory = json.load(f)

# Analyze and get critical error
results = analyze_trajectory_sync(trajectory, "your_openai_api_key")

if results['critical_error']:
    step = results['critical_error']['critical_step']
    guidance = results['critical_error']['correction_guidance']
    
    # Your correction system can:
    # 1. Reload trajectory to 'step'
    # 2. Apply correction based on 'guidance'
    # 3. Re-run from that point
```

## Core Functions

### 1. Complete Analysis (Recommended)
```python
analyze_trajectory_sync(trajectory_json: Dict, api_key: str) -> Dict
```
- **Input**: Complete trajectory JSON with messages and metadata
- **Output**: Both phase 1 errors and critical error identification
- **Use Case**: When you need full analysis

### 2. Phase 1 Only - Step-by-Step Error Detection
```python
detect_errors_sync(trajectory_json: Dict, api_key: str) -> Dict
```
- **Input**: Trajectory JSON
- **Output**: Error detection for each step and module
- **Use Case**: When you want to see all errors, not just the critical one

### 3. Phase 2 Only - Critical Error Identification
```python
find_critical_error_sync(phase1_results: Dict, trajectory_json: Dict, api_key: str) -> Dict
```
- **Input**: Phase 1 results + original trajectory
- **Output**: The earliest critical error that caused failure
- **Use Case**: When you already have Phase 1 results

## Input Format

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Your task is: put a clean plate on the table\n..."
        },
        {
            "role": "assistant", 
            "content": "<memory_analysis>...</memory_analysis><reflection>...</reflection><plan>...</plan><action>go to kitchen</action>"
        },
        {
            "role": "user",
            "content": "You are now in the kitchen..."
        }
    ],
    "metadata": {
        "task_id": "task_001",
        "won": false,
        "environment": "alfworld"
    }
}
```

## Output Format

### Critical Error (for trajectory correction)
```json
{
    "critical_step": 4,
    "critical_module": "planning",
    "error_type": "impossible_action",
    "root_cause": "The agent planned to slice a desk...",
    "evidence": "Agent output: 'slice desk with knife'",
    "correction_guidance": "The agent should recognize that desks cannot be sliced. Instead, it should look for sliceable objects like fruits or vegetables.",
    "confidence": 0.95
}
```

## Integration Example for Trajectory Correction

```python
import json
from api_interface import analyze_trajectory_sync

class TrajectoryCorrector:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def correct_trajectory(self, trajectory_path):
        # Load failed trajectory
        with open(trajectory_path, 'r') as f:
            trajectory = json.load(f)
        
        # Analyze to find critical error
        results = analyze_trajectory_sync(trajectory, self.api_key)
        
        if not results['critical_error']:
            print("No critical error found or task succeeded")
            return None
        
        critical = results['critical_error']
        
        # Extract correction information
        correction_info = {
            'reload_to_step': critical['critical_step'],
            'failed_module': critical['critical_module'],
            'error_type': critical['error_type'],
            'correction_guidance': critical['correction_guidance'],
            'original_error': critical['evidence']
        }
        
        # Your system can now:
        # 1. Reload environment to step N-1
        # 2. Apply correction based on guidance
        # 3. Continue trajectory from corrected point
        
        return correction_info

# Usage
corrector = TrajectoryCorrector("your_api_key")
correction = corrector.correct_trajectory("failed_trajectory.json")

if correction:
    print(f"Reload to step {correction['reload_to_step']}")
    print(f"Apply correction: {correction['correction_guidance']}")
```

## Error Types Reference

### Memory Errors
- `hallucination`: False memory generation
- `memory_retrieval_failure`: Failed to recall relevant information
- `over_simplification`: Lost important details in summarization

### Reflection Errors
- `progress_misassessment`: Incorrect evaluation of progress
- `outcome_misinterpretation`: Misunderstood action results
- `causal_misattribution`: Wrong cause-effect reasoning

### Planning Errors
- `impossible_action`: Planned physically/logically impossible actions
- `task_decomposition_failure`: Poor initial task breakdown (Step 1 only)
- `constraint_ignorance`: Ignored task constraints
- `inefficient_planning`: Suboptimal planning

### Action Errors
- `planning_action_disconnect`: Actions don't match plans
- `error_format`: Malformed action syntax
- `parameter_error`: Invalid action parameters

### System Errors
- `step_limit_exhaustion`: Exceeded maximum steps
- `tool_execution_error`: External tool failures
- `llm_limit`: LLM constraints
- `environment_error`: Environment bugs

## Requirements

```bash
pip install aiohttp
```

## Files Needed

1. `api_interface.py` - Main API interface (this file)
2. `analysis_v4_error_detection.py` - Phase 1 implementation
3. `analysis_phase2_v2_critical.py` - Phase 2 implementation
4. `error_type_definition.md` - Error taxonomy reference

## Support

For questions or issues, contact: Zijia Liu
# Agent Trajectory Evaluation System

A comprehensive evaluation framework for assessing agent performance across multiple cognitive modules.

## Overview

This system evaluates agent trajectories by analyzing four key cognitive modules:
1. **Memory Recall** - How well the agent remembers and uses past information
2. **Reflection** - Quality of analysis about action outcomes
3. **Thinking/Planning** - Logical reasoning and strategic planning
4. **Action Selection** - Appropriateness of chosen actions

## Features

- **5-Point Scoring Scale** (0-5) with detailed rubrics
- **Parallel Processing** for efficient evaluation of multiple trajectories
- **LLM-Based Assessment** with few-shot examples
- **Detailed Analysis** including strengths, weaknesses, and suggestions
- **Task Success Correlation** to identify failure patterns

## Installation

```bash
# Install required packages
pip install aiohttp tqdm

# Set API credentials
export OPENAI_API_KEY='your-api-key-here'
export OPENAI_API_BASE='https://api.openai.com/v1/chat/completions'  # Optional
export EVAL_MODEL='gpt-4o'  # Optional, defaults to gpt-4o
```

## Usage

### Quick Start

```bash
# Run evaluation on chat history files
python run_evaluation.py
```

### Command Line Interface

```bash
python agent_trajectory_evaluator.py \
    --input-dir trajectories/chat_histories \
    --output-dir evaluation_results \
    --max-concurrent 5 \
    --api-key YOUR_API_KEY \
    --model gpt-4o
```

### Python API

```python
import asyncio
from agent_trajectory_evaluator import EvaluationPipeline, API_CONFIG

async def evaluate_custom_trajectories():
    # Configure API
    API_CONFIG['api_key'] = 'your-api-key'
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(API_CONFIG, 'output_dir')
    
    # Process trajectories
    results = await pipeline.process_directory('input_dir')
    
    return results

# Run evaluation
results = asyncio.run(evaluate_custom_trajectories())
```

## Scoring Rubrics

### Score Interpretation

- **5 (Excellent)**: Optimal performance, sophisticated reasoning
- **4 (Good)**: Solid performance with minor issues
- **3 (Adequate)**: Acceptable but with room for improvement
- **2 (Poor)**: Significant issues affecting task progress
- **1 (Very Poor)**: Major failures, unlikely to succeed
- **0 (Failure)**: Complete failure, contradicts task goals

### Module-Specific Criteria

#### Memory Recall
- Accuracy of recalled information
- Relevance to current situation
- Learning from past attempts
- Pattern recognition

#### Reflection
- Depth of analysis
- Accuracy in interpreting outcomes
- Insight generation
- Metacognitive awareness

#### Thinking/Planning
- Logical coherence
- Strategic consideration
- Contingency planning
- Goal alignment

#### Action Selection
- Alignment with plan
- Task progress efficiency
- Edge case handling
- Error avoidance

## Output Format

### Individual Trajectory Evaluation

```json
{
  "task_id": "chat_20250829_021309",
  "task_description": "put a book in armchair",
  "success": true,
  "overall_score": 4.2,
  "summary": "Task completed successfully with strong planning...",
  "strengths": [
    "Strong Memory Recall (avg: 4.5)",
    "Strong Action Selection (avg: 4.3)"
  ],
  "weaknesses": [
    "Weak Reflection (avg: 2.1)"
  ],
  "step_evaluations": [
    {
      "step_num": 1,
      "overall_quality": 4.0,
      "critical_issues": [],
      "module_scores": {
        "memory_recall": {
          "score": 4.5,
          "reasoning": "Accurately recalls previous observations...",
          "evidence": ["I checked the coffee table and drawer 1..."],
          "suggestions": ["Consider tracking failed locations more systematically"]
        }
      }
    }
  ]
}
```

### Summary Report

```json
{
  "total_trajectories": 10,
  "average_score": 3.8,
  "success_rate": 0.7,
  "trajectories": [
    {
      "task_id": "chat_20250829_021309",
      "score": 4.2,
      "success": true
    }
  ]
}
```

## Analyzing Results

### Identifying Failure Patterns

```python
import json
from pathlib import Path

def analyze_failures(results_dir):
    """Analyze common failure patterns across trajectories"""
    
    failure_patterns = {}
    
    for eval_file in Path(results_dir).glob("eval_*.json"):
        with open(eval_file) as f:
            data = json.load(f)
        
        if not data['success']:
            # Collect weaknesses from failed trajectories
            for weakness in data['weaknesses']:
                module = weakness.split('(')[0].strip()
                failure_patterns[module] = failure_patterns.get(module, 0) + 1
    
    return failure_patterns

# Example usage
patterns = analyze_failures('evaluation_results')
print("Common failure patterns:", patterns)
```

### Module Performance Analysis

```python
def analyze_module_performance(results_dir):
    """Calculate average performance per module"""
    
    module_scores = {}
    module_counts = {}
    
    for eval_file in Path(results_dir).glob("eval_*.json"):
        with open(eval_file) as f:
            data = json.load(f)
        
        for step_eval in data['step_evaluations']:
            for module, score_data in step_eval['module_scores'].items():
                if module not in module_scores:
                    module_scores[module] = 0
                    module_counts[module] = 0
                
                module_scores[module] += score_data['score']
                module_counts[module] += 1
    
    # Calculate averages
    averages = {}
    for module in module_scores:
        averages[module] = module_scores[module] / module_counts[module]
    
    return averages
```

## Best Practices

### For Better Evaluations

1. **Ensure Clean Module Extraction**: Agent responses should use proper tags
2. **Provide Context**: Include task descriptions and observations
3. **Batch Processing**: Use appropriate concurrency limits for API calls
4. **Error Handling**: Check failed evaluations in logs

### For Training Improvements

1. **Focus on Weak Modules**: Prioritize training on consistently low-scoring modules
2. **Learn from Failures**: Analyze critical issues in failed trajectories
3. **Success Patterns**: Study high-scoring trajectories for best practices
4. **Iterative Refinement**: Re-evaluate after training improvements

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `--max-concurrent` parameter
2. **Memory Issues**: Process smaller batches of trajectories
3. **Parsing Errors**: Ensure chat history JSON format is correct
4. **Empty Evaluations**: Check if module tags are properly formatted

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
pipeline = EvaluationPipeline(API_CONFIG, output_dir)
```

## Advanced Configuration

### Custom Rubrics

```python
from agent_trajectory_evaluator import EvaluationRubrics

class CustomRubrics(EvaluationRubrics):
    MEMORY_RECALL_RUBRIC = """
    Your custom memory recall rubric here...
    """

# Use custom rubrics
evaluator.rubrics = CustomRubrics()
```

### Custom Few-Shot Examples

```python
from agent_trajectory_evaluator import FewShotExamples

class CustomExamples(FewShotExamples):
    MEMORY_EXAMPLES = [
        {
            "context": "Your task context",
            "content": "Agent's memory recall",
            "score": 5,
            "reasoning": "Why this score"
        }
    ]

# Use custom examples
evaluator.examples = CustomExamples()
```

## Performance Optimization

- **Caching**: Results are saved to avoid re-evaluation
- **Async Processing**: Utilizes asyncio for concurrent API calls
- **Batch Processing**: Process multiple files in parallel
- **Resource Management**: Automatic semaphore control for API limits

## Contributing

To improve the evaluation system:

1. Add more sophisticated rubrics
2. Expand few-shot examples
3. Implement additional analysis tools
4. Optimize API usage patterns

## License

MIT
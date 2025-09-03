#!/usr/bin/env python3
"""
VLLM Server for Reward Model
Hosts a reward model that evaluates agent trajectories based on cognitive modules.
"""

import argparse
import asyncio
from typing import Dict, List, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global engine variable
engine = None
tokenizer = None

class EvaluationRequest(BaseModel):
    """Request model for trajectory evaluation"""
    module_name: str  # planner, executor, reflection, memory_use
    trajectory: str
    task_description: str
    step_num: int
    failure_context: Dict[str, Any]

class EvaluationResponse(BaseModel):
    """Response model for trajectory evaluation"""
    score: float  # 0-1 normalized score
    raw_score: int  # 0-5 original score
    failure_type: str
    reasoning: str

class RewardModelEvaluator:
    """Evaluator that scores trajectories based on cognitive modules"""
    
    def __init__(self):
        self.module_mappings = {
            'planner': 'thinking',
            'executor': 'action', 
            'reflection': 'reflection',
            'memory_use': 'memory_recall'
        }
    
    def get_module_rubric(self, module_name: str) -> str:
        """Get scoring rubric for specific module"""
        rubrics = {
            'memory_recall': """
Memory Recall Module Scoring (0-5):
5 - Excellent: Perfect recall with strategic learning from experience
4 - Good: Recalls most relevant info with minor gaps
3 - Adequate: Basic recall but limited learning
2 - Poor: Repetitive memory without strategic value
1 - Very Poor: Inappropriate or irrelevant memory usage
0 - Failure: False memory or complete absence when needed
            """,
            'reflection': """
Reflection Module Scoring (0-5):
5 - Excellent: Deep causal analysis with actionable insights
4 - Good: Solid analysis with useful conclusions
3 - Adequate: Basic outcome recognition
2 - Poor: Shallow analysis missing key implications
1 - Very Poor: Misinterprets outcomes
0 - Failure: No reflection when needed
            """,
            'thinking': """
Planning/Thinking Module Scoring (0-5):
5 - Excellent: Clear strategic planning with contingencies
4 - Good: Solid planning with logical progression
3 - Adequate: Basic planning toward goals
2 - Poor: Vague or inconsistent planning
1 - Very Poor: Plans work against objectives
0 - Failure: No planning or incoherent plans
            """,
            'action': """
Action Module Scoring (0-5):
5 - Excellent: Optimal action aligned with plan
4 - Good: Effective action advancing task
3 - Adequate: Reasonable progress
2 - Poor: Suboptimal with questionable reasoning
1 - Very Poor: Actions hinder progress
0 - Failure: Invalid or impossible actions
            """
        }
        mapped_name = self.module_mappings.get(module_name, module_name)
        return rubrics.get(mapped_name, "")
    
    def build_evaluation_prompt(
        self,
        module_name: str,
        content: str,
        task_description: str,
        step_num: int,
        failure_context: Dict[str, Any]
    ) -> str:
        """Build evaluation prompt for the reward model"""
        
        mapped_module = self.module_mappings.get(module_name, module_name)
        rubric = self.get_module_rubric(mapped_module)
        
        failure_info = f"""
FAILURE CONTEXT FOR STEP {step_num}:
- Consecutive failures: {failure_context.get('consecutive_failures', 0)}
- Same strategy repeats: {failure_context.get('same_strategy_repeats', 0)}
- Total failures so far: {failure_context.get('total_failures', 0)}
"""
        
        prompt = f"""
You are an expert evaluator assessing agent performance.

TASK: {task_description}

{failure_info}

EVALUATION RUBRIC:
{rubric}

CONTENT TO EVALUATE:
Step {step_num} - {module_name}: {content}

EVALUATION INSTRUCTIONS:
1. Consider both content quality and failure history
2. Assign ONE integer score (0-5)
3. Provide brief reasoning

OUTPUT FORMAT:
<score>[0-5 integer]</score>
<failure_type>[type from rubric]</failure_type>
<reasoning>[brief explanation]</reasoning>
"""
        return prompt
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response to extract score and reasoning"""
        
        # Extract score
        score_match = re.search(r'<score>(\d+)</score>', response)
        score = int(score_match.group(1)) if score_match else 0
        score = max(0, min(5, score))  # Clamp to 0-5
        
        # Extract failure type
        failure_match = re.search(r'<failure_type>(.*?)</failure_type>', response, re.DOTALL)
        failure_type = failure_match.group(1).strip() if failure_match else "unknown"
        
        # Extract reasoning
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            'raw_score': score,
            'score': score / 5.0,  # Normalize to 0-1
            'failure_type': failure_type,
            'reasoning': reasoning
        }

evaluator = RewardModelEvaluator()

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_trajectory(request: EvaluationRequest):
    """Evaluate a trajectory step for a specific module"""
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Build evaluation prompt
        prompt = evaluator.build_evaluation_prompt(
            module_name=request.module_name,
            content=request.trajectory,
            task_description=request.task_description,
            step_num=request.step_num,
            failure_context=request.failure_context
        )
        
        # Generate evaluation
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=["</reasoning>"]
        )
        
        request_id = f"eval_{request.module_name}_{request.step_num}"
        results = await engine.generate(prompt, sampling_params, request_id)
        
        # Parse response
        output_text = results.outputs[0].text
        parsed = evaluator.parse_response(output_text)
        
        return EvaluationResponse(
            score=parsed['score'],
            raw_score=parsed['raw_score'],
            failure_type=parsed['failure_type'],
            reasoning=parsed['reasoning']
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": engine is not None}

async def initialize_engine(model_path: str, gpu_memory_utilization: float = 0.3):
    """Initialize VLLM engine"""
    global engine, tokenizer
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        disable_log_requests=True,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info(f"Model loaded: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="VLLM Reward Model Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to reward model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8100, help="Port number")
    parser.add_argument("--gpu-memory", type=float, default=0.3, help="GPU memory utilization")
    
    args = parser.parse_args()
    
    # Initialize engine
    asyncio.run(initialize_engine(args.model_path, args.gpu_memory))
    
    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Agent Trajectory Evaluation System

Evaluates agent trajectories by scoring four cognitive modules:
- Memory Recall
- Reflection  
- Planning/Thinking
- Action Selection

Uses LLM-based evaluation with detailed rubrics for agent training.
"""

import json
import os
import asyncio
import aiohttp
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEvaluation:
    """Complete trajectory evaluation results"""
    task_id: str
    task_description: str
    success: bool
    trajectory_text: str
    module_scores: Dict[str, Dict[str, Any]]  # module_name -> {score, reasoning, evidence, suggestions}
    overall_score: float
    critical_issues: List[str]
    recommendations: List[str]


class TrajectoryEvaluator:
    """Main evaluator for agent trajectories"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
    
    def parse_chat_history(self, file_path: str) -> Dict[str, Any]:
        """Parse chat history file into structured data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        chat_history = data.get('chat_history', [])
        
        # Extract task description
        task_description = ""
        for msg in chat_history:
            if msg['role'] == 'user' and 'task is to:' in msg['content']:
                task_match = re.search(r'Your task is to: (.+?)(?:\n|\.)', msg['content'])
                if task_match:
                    task_description = task_match.group(1).strip()
                    break
        
        # Extract module content with step annotations
        modules = {
            'memory_recall': [],
            'reflection': [],
            'thinking': [],
            'action': []
        }
        
        step_num = 0
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                step_num += 1
                content = msg['content']
                
                # Extract each module with step annotation
                for module_name in modules.keys():
                    if module_name == 'memory_recall':
                        pattern = r'<memory_recall>(.*?)</memory_recall>'
                    elif module_name == 'reflection':
                        pattern = r'<reflection>(.*?)</reflection>'
                    elif module_name == 'thinking':
                        pattern = r'<think>(.*?)</think>'
                    elif module_name == 'action':
                        pattern = r'<action>(.*?)</action>'
                    
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        module_content = match.group(1).strip()
                        if module_content:
                            modules[module_name].append({
                                'step': step_num,
                                'content': module_content
                            })
        
        return {
            'task_id': metadata.get('timestamp', 'unknown'),
            'task_description': task_description,
            'success': metadata.get('success', False),
            'trajectory_text': json.dumps(chat_history, indent=2),
            'modules': modules,
            'metadata': metadata
        }
    
    async def evaluate_trajectory(self, trajectory_data: Dict[str, Any]) -> TrajectoryEvaluation:
        """Evaluate complete trajectory"""
        
        module_scores = {}
        
        # Evaluate each module
        for module_name, module_instances in trajectory_data['modules'].items():
            if module_instances:
                score_data = await self.evaluate_module(
                    module_name,
                    module_instances,
                    trajectory_data
                )
                module_scores[module_name] = score_data
        
        # Calculate overall score
        scores = [data['score'] for data in module_scores.values()]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Identify critical issues and recommendations
        critical_issues = []
        recommendations = []
        
        for module_name, data in module_scores.items():
            if data['score'] <= 1:
                critical_issues.append(f"Critical failure in {module_name}: {data['reasoning'][:100]}...")
            if data['score'] <= 2:
                recommendations.extend(data.get('suggestions', []))
        
        return TrajectoryEvaluation(
            task_id=trajectory_data['task_id'],
            task_description=trajectory_data['task_description'],
            success=trajectory_data['success'],
            trajectory_text=trajectory_data['trajectory_text'],
            module_scores=module_scores,
            overall_score=overall_score,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    async def evaluate_module(
        self,
        module_name: str,
        module_instances: List[Dict[str, Any]],
        trajectory_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a specific module across the entire trajectory"""
        
        prompt = self.build_module_prompt(module_name, module_instances, trajectory_data)
        response = await self.call_llm(prompt)
        return self.parse_evaluation_response(response)
    
    def build_module_prompt(
        self,
        module_name: str,
        module_instances: List[Dict[str, Any]],
        trajectory_data: Dict[str, Any]
    ) -> str:
        """Build evaluation prompt for a specific module"""
        
        rubrics = self.get_rubrics()
        examples = self.get_examples()
        
        # Format module instances
        instances_text = "\n".join([
            f"Step {inst['step']}: {inst['content']}"
            for inst in module_instances
        ])
        
        prompt = f"""
You are an expert evaluator assessing agent performance in autonomous task execution.
Evaluate the {module_name.replace('_', ' ').title()} module across the entire trajectory.

TASK CONTEXT:
Task: {trajectory_data['task_description']}
Task Success: {trajectory_data['success']}
Total Steps: {len(module_instances)}

EVALUATION RUBRIC:
{rubrics[module_name]}

FEW-SHOT EXAMPLES:
{examples[module_name]}

MODULE INSTANCES TO EVALUATE:
{instances_text}

EVALUATION INSTRUCTIONS:
1. Assess the module's performance across the entire trajectory
2. Consider consistency, improvement over time, and task alignment
3. Identify specific strengths and critical weaknesses
4. Provide actionable suggestions for improvement

OUTPUT FORMAT (JSON):
{{
    "score": [0-5 integer],
    "reasoning": "Detailed explanation of score with specific evidence",
    "evidence": ["Specific quote from Step X", "Another evidence from Step Y"],
    "suggestions": ["Concrete improvement 1", "Concrete improvement 2"],
    "error_types": ["Error type 1", "Error type 2"],
    "consistency_score": [0-5 integer]
}}

Provide your evaluation:
"""
        return prompt
    
    def get_rubrics(self) -> Dict[str, str]:
        """Get detailed scoring rubrics for each module"""
        return {
            'memory_recall': """
Memory Recall Module Scoring (0-5):

5 - Excellent:
- Accurately recalls all relevant past experiences, observations, and outcomes
- Makes sophisticated connections between past and current situations
- Demonstrates clear learning from previous attempts and failures
- Tracks progress systematically and updates understanding
- Uses memory to avoid repeating mistakes

4 - Good:
- Recalls most relevant information with minor gaps
- Makes useful connections to guide decisions
- Shows awareness of what has been tried before
- Generally learns from experience

3 - Adequate:
- Recalls basic relevant information
- Some connections made between past and present
- Limited evidence of learning from failures
- Inconsistent memory utilization

2 - Poor:
- Significant gaps in recall or inaccuracies
- Fails to leverage important past experiences
- Repeats previously failed approaches without adaptation
- Poor pattern recognition

1 - Very Poor:
- Major recall failures, contradictions, or fabrications
- Ignores critical past observations
- No evidence of learning from repeated failures
- Consistently forgets important context

0 - Failure:
- No relevant recall or completely inaccurate information
- Contradicts well-established facts from the trajectory
- Shows no memory of previous steps or outcomes

Common Error Types:
- Repetition without learning: Continuing failed strategies
- Selective memory: Only remembering successes, ignoring failures
- False memories: Claiming things that didn't happen
- Context loss: Forgetting task objectives or constraints
- Pattern blindness: Missing obvious repeated patterns
            """,
            
            'reflection': """
Reflection Module Scoring (0-5):

5 - Excellent:
- Deep, insightful analysis of action outcomes and their implications
- Accurately identifies both successes and failures with specific reasons
- Extracts meaningful insights that inform future strategy
- Shows metacognitive awareness and self-correction
- Analyzes not just what happened but why it happened

4 - Good:
- Solid analysis of outcomes with good understanding
- Recognizes key successes and failures accurately
- Generates useful insights for future actions
- Some metacognitive awareness

3 - Adequate:
- Basic reflection on obvious outcomes
- Identifies clear successes/failures
- Limited depth in causal analysis
- Some useful insights generated

2 - Poor:
- Superficial reflection missing important implications
- Misses key outcomes or their significance
- Little insight generated for future improvement
- Poor causal understanding

1 - Very Poor:
- Minimal reflection with significant misunderstandings
- Misinterprets clear outcomes
- No useful insights or learning
- May blame external factors inappropriately

0 - Failure:
- No meaningful reflection or completely wrong interpretation
- Contradicts observable outcomes
- Shows no understanding of cause and effect

Common Error Types:
- Outcome misinterpretation: Wrong understanding of what happened
- Shallow analysis: Only surface-level observations
- Blame shifting: Attributing failures to environment vs strategy
- Success fixation: Only reflecting on successes, ignoring failures
- No causal reasoning: Describing what without understanding why
            """,
            
            'thinking': """
Planning/Thinking Module Scoring (0-5):

5 - Excellent:
- Clear, logical, step-by-step planning with detailed reasoning
- Considers multiple approaches and evaluates trade-offs
- Anticipates potential obstacles and has contingency plans
- Directly addresses task requirements and constraints
- Shows strategic thinking and resource optimization

4 - Good:
- Solid planning with clear logical progression
- Considers main options and their implications
- Good alignment with task objectives
- Some strategic consideration

3 - Adequate:
- Basic planning present with some logical structure
- Generally moves toward stated goals
- Limited consideration of alternatives
- Addresses immediate needs

2 - Poor:
- Weak or inconsistent planning
- Limited logical reasoning
- May not effectively advance toward task completion
- Poor consideration of constraints

1 - Very Poor:
- Confused, contradictory, or illogical planning
- Poor reasoning that may hinder progress
- Misaligned with task objectives
- No systematic approach

0 - Failure:
- No coherent planning or completely illogical
- Actively works against task objectives
- Contradicts established information

Common Error Types:
- Goal drift: Losing sight of main objective
- No systematic approach: Random or reactive planning
- Constraint ignorance: Planning without considering limitations
- Over-complexity: Unnecessarily complicated approaches
- Under-planning: Insufficient consideration of steps needed
            """,
            
            'action': """
Action Module Scoring (0-5):

5 - Excellent:
- Optimal action choice given current context and constraints
- Perfect alignment with stated plan and task objectives
- Efficient progress toward task completion
- Handles edge cases and unexpected situations appropriately
- Shows good judgment in action selection

4 - Good:
- Good action choice that advances task effectively
- Well-aligned with plan and objectives
- Makes solid progress with minor inefficiencies
- Generally appropriate responses

3 - Adequate:
- Reasonable action choice that makes some progress
- Generally aligned with goals
- May not be optimal but acceptable
- Some progress toward completion

2 - Poor:
- Suboptimal action choice with limited effectiveness
- Poor alignment with stated plan or objectives
- Inefficient progress or creates additional problems
- May require significant correction

1 - Very Poor:
- Poor action choice that hinders progress
- Misaligned with plan and objectives
- Creates problems or moves away from goal
- Shows poor judgment

0 - Failure:
- Completely wrong action that contradicts objectives
- Actively works against task completion
- Violates constraints or safety requirements

Common Error Types:
- Plan deviation: Actions not matching stated intentions
- Inefficient exploration: Poor search strategies
- Constraint violation: Actions not allowed in context
- Goal confusion: Actions serving wrong objectives
- Persistence errors: Repeating failed actions without adaptation
            """
        }
    
    def get_examples(self) -> Dict[str, str]:
        """Get few-shot examples for each module"""
        return {
            'memory_recall': """
Example 1 (Score: 5):
Context: Multi-step cooking task
Content: "I previously found flour in cabinet 2 and eggs in the refrigerator. The recipe called for 2 cups flour and 3 eggs, which I have located. I still need to find sugar and vanilla extract. I remember seeing spice containers in cabinet 4, so vanilla might be there. I haven't checked the pantry yet for sugar."
Reasoning: Excellent recall of specific locations, quantities, and remaining requirements. Shows systematic tracking of progress.

Example 2 (Score: 1):
Context: Navigation task after multiple failed attempts
Content: "I think I need to go somewhere."
Reasoning: Completely vague, no recall of previous attempts, destinations tried, or lessons learned.

Example 3 (Score: 3):
Context: Object manipulation task
Content: "I tried to pick up the object before but it didn't work. I should try a different approach."
Reasoning: Basic recall of failure but lacks specifics about why it failed or what approaches were tried.
            """,
            
            'reflection': """
Example 1 (Score: 5):
Context: Failed to unlock door with attempted key
Content: "The key didn't work because I can see it's actually a car key with a different shape than door keys. This explains the failure - I was using the wrong tool entirely. I need to search for a house key specifically, probably in key holders or drawers where such items are typically stored."
Reasoning: Identifies specific cause, draws logical conclusions, provides clear next steps.

Example 2 (Score: 1):
Context: Successfully completed subtask
Content: "Good."
Reasoning: No analysis, no insights, minimal engagement with the outcome.

Example 3 (Score: 2):
Context: Failed action sequence
Content: "It didn't work. Maybe the environment is broken."
Reasoning: Acknowledges failure but immediately blames external factors without self-analysis.
            """,
            
            'thinking': """
Example 1 (Score: 5):
Context: Complex assembly task
Content: "I need to assemble this bookshelf systematically. First, I'll organize all pieces and identify them using the instruction manual. Then I'll gather necessary tools - I saw a screwdriver in the toolbox earlier. I'll start with the base frame as it provides stability for subsequent pieces. If I encounter missing pieces, I'll check the packaging again before proceeding."
Reasoning: Systematic approach, considers prerequisites, has contingency planning, logical sequence.

Example 2 (Score: 2):
Context: Search and retrieval task  
Content: "I'll look around and find it somehow."
Reasoning: Vague, no specific plan, no systematic approach, unlikely to be effective.

Example 3 (Score: 4):
Context: Cleaning task
Content: "I'll start with the kitchen since it's typically the messiest area, then move to living areas. I should gather cleaning supplies first from the utility closet."
Reasoning: Good prioritization and logical sequence, though could be more detailed.
            """,
            
            'action': """
Example 1 (Score: 5):
Context: Plan was to check kitchen cabinet for cooking utensils
Content: "action_choice: 'go to kitchen'\naction_parameters: {}"
Reasoning: Perfect alignment with plan, appropriate action for stated objective.

Example 2 (Score: 0):
Context: Plan was to find books in library, agent is currently in library
Content: "action_choice: 'go to garage'\naction_parameters: {}"
Reasoning: Completely contradicts plan and current context, wrong location entirely.

Example 3 (Score: 3):
Context: Searching for keys, plan was to check drawers systematically
Content: "action_choice: 'go to drawer 3'\naction_parameters: {}"
Reasoning: Aligns with plan but skips drawer 1 and 2 without explanation, somewhat inefficient.
            """
        }
    
    async def call_llm(self, prompt: str) -> str:
        """Make API call to LLM"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "system", "content": "You are an expert evaluator of autonomous agent performance. Provide precise, evidence-based assessments."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config['temperature']
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config['max_retries']):
                try:
                    async with session.post(
                        self.config['base_url'],
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config['timeout'])
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                except Exception as e:
                    if attempt == self.config['max_retries'] - 1:
                        logger.error(f"API call failed after {self.config['max_retries']} attempts: {e}")
                        raise e
                    await asyncio.sleep(2 ** attempt)
        
        return ""
    
    def parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM evaluation response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    'score': float(data.get('score', 0)),
                    'reasoning': data.get('reasoning', ''),
                    'evidence': data.get('evidence', []),
                    'suggestions': data.get('suggestions', []),
                    'error_types': data.get('error_types', []),
                    'consistency_score': float(data.get('consistency_score', 0))
                }
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # Fallback parsing
        score_match = re.search(r'score["\s:]+(\d+)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return {
            'score': score,
            'reasoning': response,
            'evidence': [],
            'suggestions': [],
            'error_types': [],
            'consistency_score': score
        }
    
    async def process_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single trajectory file"""
        try:
            # Parse trajectory
            trajectory_data = self.parse_chat_history(file_path)
            
            # Evaluate trajectory
            evaluation = await self.evaluate_trajectory(trajectory_data)
            
            # Prepare output
            output = {
                'file_path': file_path,
                'task_id': evaluation.task_id,
                'task_description': evaluation.task_description,
                'success': evaluation.success,
                'overall_score': evaluation.overall_score,
                'module_scores': evaluation.module_scores,
                'critical_issues': evaluation.critical_issues,
                'recommendations': evaluation.recommendations,
                'trajectory_text': evaluation.trajectory_text,
                'timestamp': str(datetime.now())
            }
            
            # Save individual result
            output_file = Path(output_dir) / f"eval_{Path(file_path).stem}.json"
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            logger.info(f"Evaluated {evaluation.task_id}: Score {evaluation.overall_score:.2f}")
            return output
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    async def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process all trajectory files in directory"""
        
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {input_dir}")
            return []
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process files with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(file_path):
            async with semaphore:
                return await self.process_file(str(file_path), output_dir)
        
        tasks = [process_with_limit(f) for f in json_files]
        
        results = []
        with tqdm(total=len(tasks), desc="Evaluating trajectories") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)
                pbar.update(1)
        
        # Generate summary
        self.generate_summary(results, output_dir)
        
        return results
    
    def generate_summary(self, results: List[Dict[str, Any]], output_dir: str):
        """Generate evaluation summary"""
        
        if not results:
            return
        
        # Calculate statistics
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        scores = [r['overall_score'] for r in results]
        
        # Module performance
        module_stats = {}
        for module_name in ['memory_recall', 'reflection', 'thinking', 'action']:
            module_scores = []
            error_types = []
            
            for result in results:
                if module_name in result['module_scores']:
                    module_data = result['module_scores'][module_name]
                    module_scores.append(module_data['score'])
                    error_types.extend(module_data.get('error_types', []))
            
            if module_scores:
                module_stats[module_name] = {
                    'average_score': sum(module_scores) / len(module_scores),
                    'total_evaluations': len(module_scores),
                    'common_errors': list(set(error_types))
                }
        
        # Create summary
        summary = {
            'evaluation_timestamp': str(datetime.now()),
            'overview': {
                'total_trajectories': total,
                'successful_trajectories': successful,
                'success_rate': successful / total,
                'average_overall_score': sum(scores) / len(scores),
                'score_distribution': {
                    'excellent (4-5)': sum(1 for s in scores if s >= 4),
                    'good (3-4)': sum(1 for s in scores if 3 <= s < 4),
                    'poor (1-3)': sum(1 for s in scores if 1 <= s < 3),
                    'failure (0-1)': sum(1 for s in scores if s < 1)
                }
            },
            'module_performance': module_stats,
            'worst_performers': sorted(results, key=lambda x: x['overall_score'])[:5],
            'best_performers': sorted(results, key=lambda x: x['overall_score'], reverse=True)[:5]
        }
        
        # Save summary
        summary_file = Path(output_dir) / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TRAJECTORY EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trajectories: {total}")
        print(f"Success Rate: {successful}/{total} ({successful/total:.1%})")
        print(f"Average Score: {sum(scores)/len(scores):.2f}/5.0")
        print(f"\nModule Performance:")
        for module, stats in module_stats.items():
            print(f"  {module.replace('_', ' ').title()}: {stats['average_score']:.2f}/5.0")
        print(f"\nResults saved to: {output_dir}")


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Evaluate agent trajectories")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing trajectory JSON files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--api-url", type=str, default="https://api.openai.com/v1/chat/completions", help="API URL")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent evaluations")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    
    args = parser.parse_args()
    
    # Configure API
    api_config = {
        "base_url": args.api_url,
        "api_key": args.api_key or os.getenv("OPENAI_API_KEY", ""),
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": 3,
        "timeout": 60
    }
    
    if not api_config["api_key"]:
        print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        return
    
    # Initialize evaluator
    evaluator = TrajectoryEvaluator(api_config)
    
    # Process trajectories
    await evaluator.process_directory(
        args.input_dir,
        args.output_dir,
        args.max_concurrent
    )


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())
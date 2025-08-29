"""
Agent Trajectory Evaluation System

A comprehensive evaluation framework for assessing agent trajectories across different modules:
- Memory Recall
- Reflection
- Planning/Thinking
- Action Selection

The system uses LLM-based evaluation with detailed rubrics and few-shot examples.
"""

import json
import os
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm


# ====== Configuration ======
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", ""),
    "api_key": os.getenv("API_KEY", ""),
    "model": os.getenv("MODEL", "gpt-4o"),
    "temperature": 0.0,
    "max_retries": 3,
    "timeout": 60
}


# ====== Data Classes ======
@dataclass
class ModuleContent:
    """Container for extracted module content from agent response"""
    memory_recall: str = ""
    reflection: str = ""
    thinking: str = ""
    action: str = ""
    raw_response: str = ""


@dataclass
class StepData:
    """Container for a single step in trajectory"""
    step_num: int
    observation: str
    modules: ModuleContent
    metadata: Dict[str, Any] = None


@dataclass
class TrajectoryData:
    """Container for complete trajectory data"""
    task_id: str
    task_description: str
    success: bool
    steps: List[StepData]
    metadata: Dict[str, Any] = None


@dataclass
class ModuleScore:
    """Container for module evaluation results"""
    module_name: str
    score: float  # 0-5 scale
    reasoning: str
    evidence: List[str]
    suggestions: List[str]


@dataclass
class StepEvaluation:
    """Container for step-level evaluation results"""
    step_num: int
    module_scores: Dict[str, ModuleScore]
    overall_quality: float
    critical_issues: List[str]


@dataclass
class TrajectoryEvaluation:
    """Container for trajectory-level evaluation results"""
    task_id: str
    task_description: str
    success: bool
    step_evaluations: List[StepEvaluation]
    overall_score: float
    summary: str
    strengths: List[str]
    weaknesses: List[str]


# ====== Module Extractors ======
class ModuleExtractor:
    """Extract module content from agent responses"""
    
    PATTERNS = {
        'memory_recall': r'<memory_recall>(.*?)</memory_recall>',
        'reflection': r'<reflection>(.*?)</reflection>',
        'thinking': r'<think>(.*?)</think>',
        'action': r'<action>(.*?)</action>'
    }
    
    @classmethod
    def extract_modules(cls, response: str) -> ModuleContent:
        """Extract all module content from agent response"""
        modules = ModuleContent(raw_response=response)
        
        for module_name, pattern in cls.PATTERNS.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if module_name == 'thinking':
                    modules.thinking = content
                else:
                    setattr(modules, module_name, content)
        
        return modules


# ====== Trajectory Parser ======
class TrajectoryParser:
    """Parse chat history into structured trajectory data"""
    
    @classmethod
    def parse_chat_history(cls, chat_history_path: str) -> TrajectoryData:
        """Parse a chat history JSON file into trajectory data"""
        with open(chat_history_path, 'r') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        chat_history = data.get('chat_history', [])
        
        # Extract task description from first user message
        task_description = ""
        for msg in chat_history:
            if msg['role'] == 'user' and 'task is to:' in msg['content']:
                task_match = re.search(r'Your task is to: (.+?)(?:\n|$)', msg['content'])
                if task_match:
                    task_description = task_match.group(1)
                    break
        
        # Parse steps from chat history
        steps = []
        step_num = 0
        
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                step_num += 1
                
                # Get corresponding observation from previous user message
                observation = ""
                if i > 0 and chat_history[i-1]['role'] == 'user':
                    observation = chat_history[i-1]['content']
                
                # Extract modules from assistant response
                modules = ModuleExtractor.extract_modules(msg['content'])
                
                step = StepData(
                    step_num=step_num,
                    observation=observation,
                    modules=modules,
                    metadata={'message_index': i}
                )
                steps.append(step)
        
        return TrajectoryData(
            task_id=metadata.get('timestamp', 'unknown'),
            task_description=task_description,
            success=metadata.get('success', False),
            steps=steps,
            metadata=metadata
        )


# ====== Evaluation Rubrics ======
class EvaluationRubrics:
    """Detailed scoring rubrics for each module"""
    
    MEMORY_RECALL_RUBRIC = """
    Memory Recall Module Scoring (0-5 scale):
    
    5 - Excellent: 
    - Accurately recalls all relevant past experiences and observations
    - Makes insightful connections between past and current situations
    - Identifies patterns and learns from previous attempts
    - Demonstrates clear understanding of task progress
    
    4 - Good:
    - Recalls most relevant information accurately
    - Makes some connections to guide current decisions
    - Shows awareness of what has been tried before
    
    3 - Adequate:
    - Recalls basic relevant information
    - Some minor inaccuracies or omissions
    - Limited connection-making between experiences
    
    2 - Poor:
    - Significant gaps in recall or inaccuracies
    - Fails to leverage important past experiences
    - Repeats previously failed approaches without learning
    
    1 - Very Poor:
    - Major recall failures or contradictions
    - Ignores critical past observations
    - No evidence of learning from experience
    
    0 - Failure:
    - No relevant recall or completely inaccurate
    - Contradicts established facts
    - Shows no memory of previous steps
    """
    
    REFLECTION_RUBRIC = """
    Reflection Module Scoring (0-5 scale):
    
    5 - Excellent:
    - Deep analysis of action outcomes and their implications
    - Identifies both successes and failures accurately
    - Extracts meaningful insights for future actions
    - Shows metacognitive awareness
    
    4 - Good:
    - Solid analysis of what happened and why
    - Recognizes key outcomes and their significance
    - Some useful insights generated
    
    3 - Adequate:
    - Basic reflection on action outcomes
    - Identifies obvious successes/failures
    - Limited depth in analysis
    
    2 - Poor:
    - Superficial or incomplete reflection
    - Misses important outcomes or implications
    - Little insight generated
    
    1 - Very Poor:
    - Minimal reflection with significant gaps
    - Misinterprets outcomes
    - No useful insights
    
    0 - Failure:
    - No meaningful reflection
    - Completely misunderstands situation
    - Contradicts observable outcomes
    """
    
    THINKING_RUBRIC = """
    Planning/Thinking Module Scoring (0-5 scale):
    
    5 - Excellent:
    - Clear, logical, step-by-step planning
    - Considers multiple options and trade-offs
    - Anticipates potential issues and has contingencies
    - Directly addresses task requirements
    
    4 - Good:
    - Solid planning with clear reasoning
    - Considers main options
    - Good alignment with task goals
    
    3 - Adequate:
    - Basic planning present
    - Some logical reasoning
    - Generally moves toward goal
    
    2 - Poor:
    - Weak or illogical planning
    - Limited consideration of options
    - May not effectively advance task
    
    1 - Very Poor:
    - Confused or contradictory planning
    - Poor reasoning
    - Unlikely to achieve goals
    
    0 - Failure:
    - No coherent planning
    - Completely illogical
    - Works against task objectives
    """
    
    ACTION_RUBRIC = """
    Action Module Scoring (0-5 scale):
    
    5 - Excellent:
    - Optimal action choice given context
    - Perfectly aligned with plan and goals
    - Efficient progress toward task completion
    - Handles edge cases appropriately
    
    4 - Good:
    - Good action choice
    - Well-aligned with plan
    - Makes solid progress
    
    3 - Adequate:
    - Reasonable action choice
    - Some progress toward goal
    - May not be optimal but acceptable
    
    2 - Poor:
    - Suboptimal action choice
    - Limited progress or inefficient
    - May require correction later
    
    1 - Very Poor:
    - Poor action choice
    - Little to no progress
    - Creates additional problems
    
    0 - Failure:
    - Completely wrong action
    - Works against task goals
    - May cause task failure
    """


# ====== Few-Shot Examples ======
class FewShotExamples:
    """Complex few-shot examples for evaluation"""
    
    MEMORY_EXAMPLES = [
        {
            "context": "Task: Find and deliver a package to room 305",
            "content": "I previously checked rooms 301, 302, and 303 on the third floor. The package was not in any of those rooms. I also recall seeing a delivery note on the reception desk mentioning packages are typically stored in the mailroom on the first floor. I haven't checked there yet.",
            "score": 5,
            "reasoning": "Excellent recall of specific rooms checked, remembers relevant detail about delivery note, identifies unexplored area logically"
        },
        {
            "context": "Task: Cook pasta with tomato sauce",
            "content": "I found the pasta in the pantry earlier. I think I saw something red in the fridge.",
            "score": 2,
            "reasoning": "Vague recall ('something red'), missing critical details about cooking steps taken, no mention of water boiling or other preparations"
        }
    ]
    
    REFLECTION_EXAMPLES = [
        {
            "context": "Previous action: Attempted to open locked door with wrong key",
            "content": "The key didn't work because it's the wrong type - I noticed it's a car key, not a door key. This suggests I need to search for the correct key elsewhere, possibly in the key holder I saw earlier. The door remains locked, so my task is not yet complete.",
            "score": 5,
            "reasoning": "Identifies specific reason for failure, draws logical conclusion, suggests concrete next steps"
        },
        {
            "context": "Previous action: Successfully turned on computer",
            "content": "The computer is on now.",
            "score": 1,
            "reasoning": "Minimal reflection, states obvious fact without analysis or insights"
        }
    ]
    
    THINKING_EXAMPLES = [
        {
            "context": "Task: Water all plants in the house",
            "content": "I need to systematically check each room for plants. I'll start with the living room, then move clockwise through the house. I should first locate a watering can or container. Based on my observation, there's a kitchen nearby where I'm likely to find water and a suitable container. After getting water, I'll proceed room by room, keeping track of which plants I've watered to avoid duplicates.",
            "score": 5,
            "reasoning": "Systematic approach, considers prerequisites, has clear plan with logical progression"
        },
        {
            "context": "Task: Fix the broken lamp",
            "content": "I'll try to fix it somehow. Maybe check what's wrong.",
            "score": 2,
            "reasoning": "Vague planning, no specific steps, lacks systematic approach"
        }
    ]
    
    ACTION_EXAMPLES = [
        {
            "context": "Plan: Get water container from kitchen first",
            "content": "action_choice: 'go to kitchen'\naction_parameters: {}",
            "score": 5,
            "reasoning": "Action perfectly aligns with stated plan, appropriate for current goal"
        },
        {
            "context": "Plan: Find the red book on the shelf",
            "content": "action_choice: 'go to bathroom'\naction_parameters: {}",
            "score": 0,
            "reasoning": "Action completely contradicts plan, wrong location for finding books"
        }
    ]


# ====== LLM Evaluator ======
class LLMEvaluator:
    """Handles LLM-based evaluation of modules"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
    
    async def evaluate_module(
        self,
        module_name: str,
        module_content: str,
        context: Dict[str, Any],
        rubric: str,
        examples: List[Dict[str, Any]]
    ) -> ModuleScore:
        """Evaluate a single module using LLM"""
        
        prompt = self._build_evaluation_prompt(
            module_name, module_content, context, rubric, examples
        )
        
        response = await self._call_llm(prompt)
        return self._parse_evaluation_response(module_name, response)
    
    def _build_evaluation_prompt(
        self,
        module_name: str,
        module_content: str,
        context: Dict[str, Any],
        rubric: str,
        examples: List[Dict[str, Any]]
    ) -> str:
        """Build evaluation prompt for a specific module"""
        
        examples_text = "\n\n".join([
            f"Example {i+1}:\nContext: {ex['context']}\nContent: {ex['content']}\n"
            f"Score: {ex['score']}/5\nReasoning: {ex['reasoning']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"""
You are an expert evaluator assessing agent performance in task execution.
Evaluate the {module_name} module based on the provided rubric and examples.

TASK CONTEXT:
Task: {context['task_description']}
Current Step: {context['step_num']}
Previous Observation: {context['observation'][:500]}...
Task Success: {context['success']}

SCORING RUBRIC:
{rubric}

FEW-SHOT EXAMPLES:
{examples_text}

MODULE CONTENT TO EVALUATE:
{module_content if module_content else "[Empty/Missing Content]"}

EVALUATION INSTRUCTIONS:
1. Analyze the module content in the context of the task
2. Apply the scoring rubric strictly
3. Provide specific evidence for your score
4. Suggest concrete improvements

OUTPUT FORMAT (JSON):
{{
    "score": [0-5],
    "reasoning": "Detailed explanation of score",
    "evidence": ["Specific quote or observation 1", "Specific quote 2"],
    "suggestions": ["Improvement suggestion 1", "Improvement suggestion 2"]
}}

Provide your evaluation:
"""
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "system", "content": "You are an expert at evaluating agent trajectories."},
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
                        raise e
                    await asyncio.sleep(2 ** attempt)
        
        return ""
    
    def _parse_evaluation_response(self, module_name: str, response: str) -> ModuleScore:
        """Parse LLM response into ModuleScore"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ModuleScore(
                    module_name=module_name,
                    score=float(data.get('score', 0)),
                    reasoning=data.get('reasoning', ''),
                    evidence=data.get('evidence', []),
                    suggestions=data.get('suggestions', [])
                )
        except:
            pass
        
        # Fallback parsing
        score_match = re.search(r'score["\s:]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0
        
        return ModuleScore(
            module_name=module_name,
            score=score,
            reasoning=response,
            evidence=[],
            suggestions=[]
        )


# ====== Trajectory Evaluator ======
class TrajectoryEvaluator:
    """Main evaluator for complete trajectories"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.llm_evaluator = LLMEvaluator(api_config)
        self.rubrics = EvaluationRubrics()
        self.examples = FewShotExamples()
    
    async def evaluate_trajectory(self, trajectory: TrajectoryData) -> TrajectoryEvaluation:
        """Evaluate complete trajectory"""
        
        step_evaluations = []
        
        for step in trajectory.steps:
            step_eval = await self.evaluate_step(
                step, 
                trajectory.task_description,
                trajectory.success
            )
            step_evaluations.append(step_eval)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(step_evaluations)
        summary = self._generate_summary(trajectory, step_evaluations)
        strengths, weaknesses = self._identify_strengths_weaknesses(step_evaluations)
        
        return TrajectoryEvaluation(
            task_id=trajectory.task_id,
            task_description=trajectory.task_description,
            success=trajectory.success,
            step_evaluations=step_evaluations,
            overall_score=overall_score,
            summary=summary,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    async def evaluate_step(
        self, 
        step: StepData,
        task_description: str,
        task_success: bool
    ) -> StepEvaluation:
        """Evaluate a single step"""
        
        context = {
            'task_description': task_description,
            'step_num': step.step_num,
            'observation': step.observation,
            'success': task_success
        }
        
        # Evaluate each module
        module_scores = {}
        
        # Memory Recall
        if step.modules.memory_recall:
            module_scores['memory_recall'] = await self.llm_evaluator.evaluate_module(
                'memory_recall',
                step.modules.memory_recall,
                context,
                self.rubrics.MEMORY_RECALL_RUBRIC,
                self.examples.MEMORY_EXAMPLES
            )
        
        # Reflection
        if step.modules.reflection:
            module_scores['reflection'] = await self.llm_evaluator.evaluate_module(
                'reflection',
                step.modules.reflection,
                context,
                self.rubrics.REFLECTION_RUBRIC,
                self.examples.REFLECTION_EXAMPLES
            )
        
        # Thinking/Planning
        if step.modules.thinking:
            module_scores['thinking'] = await self.llm_evaluator.evaluate_module(
                'thinking',
                step.modules.thinking,
                context,
                self.rubrics.THINKING_RUBRIC,
                self.examples.THINKING_EXAMPLES
            )
        
        # Action
        if step.modules.action:
            module_scores['action'] = await self.llm_evaluator.evaluate_module(
                'action',
                step.modules.action,
                context,
                self.rubrics.ACTION_RUBRIC,
                self.examples.ACTION_EXAMPLES
            )
        
        # Calculate step quality
        overall_quality = sum(s.score for s in module_scores.values()) / max(len(module_scores), 1)
        
        # Identify critical issues
        critical_issues = []
        for name, score in module_scores.items():
            if score.score <= 1:
                critical_issues.append(f"Critical failure in {name}: {score.reasoning[:100]}...")
        
        return StepEvaluation(
            step_num=step.step_num,
            module_scores=module_scores,
            overall_quality=overall_quality,
            critical_issues=critical_issues
        )
    
    def _calculate_overall_score(self, step_evaluations: List[StepEvaluation]) -> float:
        """Calculate overall trajectory score"""
        if not step_evaluations:
            return 0.0
        
        total_score = sum(eval.overall_quality for eval in step_evaluations)
        return total_score / len(step_evaluations)
    
    def _generate_summary(
        self, 
        trajectory: TrajectoryData,
        step_evaluations: List[StepEvaluation]
    ) -> str:
        """Generate summary of trajectory evaluation"""
        
        avg_score = self._calculate_overall_score(step_evaluations)
        critical_steps = [e for e in step_evaluations if e.critical_issues]
        
        summary = f"Task: {trajectory.task_description}\n"
        summary += f"Success: {trajectory.success}\n"
        summary += f"Average Score: {avg_score:.2f}/5\n"
        summary += f"Total Steps: {len(step_evaluations)}\n"
        summary += f"Critical Issues in {len(critical_steps)} steps\n"
        
        if not trajectory.success and critical_steps:
            summary += "\nLikely failure causes:\n"
            for step in critical_steps[:3]:  # Top 3 critical steps
                summary += f"- Step {step.step_num}: {', '.join(step.critical_issues[:1])}\n"
        
        return summary
    
    def _identify_strengths_weaknesses(
        self,
        step_evaluations: List[StepEvaluation]
    ) -> Tuple[List[str], List[str]]:
        """Identify trajectory strengths and weaknesses"""
        
        strengths = []
        weaknesses = []
        
        # Aggregate module scores
        module_totals = {}
        module_counts = {}
        
        for eval in step_evaluations:
            for name, score in eval.module_scores.items():
                if name not in module_totals:
                    module_totals[name] = 0
                    module_counts[name] = 0
                module_totals[name] += score.score
                module_counts[name] += 1
        
        # Identify strengths and weaknesses
        for name in module_totals:
            avg = module_totals[name] / module_counts[name]
            if avg >= 4:
                strengths.append(f"Strong {name.replace('_', ' ').title()} (avg: {avg:.1f})")
            elif avg <= 2:
                weaknesses.append(f"Weak {name.replace('_', ' ').title()} (avg: {avg:.1f})")
        
        return strengths, weaknesses


# ====== Main Pipeline ======
class EvaluationPipeline:
    """Main pipeline for processing multiple trajectories"""
    
    def __init__(self, api_config: Dict[str, Any], output_dir: str):
        self.evaluator = TrajectoryEvaluator(api_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_trajectory_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single trajectory file"""
        
        # Parse trajectory
        trajectory = TrajectoryParser.parse_chat_history(file_path)
        
        # Evaluate trajectory
        evaluation = await self.evaluator.evaluate_trajectory(trajectory)
        
        # Prepare output
        output = {
            'file_path': file_path,
            'task_id': evaluation.task_id,
            'task_description': evaluation.task_description,
            'success': evaluation.success,
            'overall_score': evaluation.overall_score,
            'summary': evaluation.summary,
            'strengths': evaluation.strengths,
            'weaknesses': evaluation.weaknesses,
            'step_evaluations': [
                {
                    'step_num': step_eval.step_num,
                    'overall_quality': step_eval.overall_quality,
                    'critical_issues': step_eval.critical_issues,
                    'module_scores': {
                        name: {
                            'score': score.score,
                            'reasoning': score.reasoning,
                            'evidence': score.evidence,
                            'suggestions': score.suggestions
                        }
                        for name, score in step_eval.module_scores.items()
                    }
                }
                for step_eval in evaluation.step_evaluations
            ]
        }
        
        # Save to file
        output_file = self.output_dir / f"eval_{Path(file_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output
    
    async def process_directory(self, input_dir: str, max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Process all trajectory files in directory with concurrency control"""
        
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))
        
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_limit(file_path):
            async with semaphore:
                try:
                    return await self.process_trajectory_file(str(file_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    return None
        
        tasks = [process_with_limit(f) for f in json_files]
        
        with tqdm(total=len(tasks), desc="Evaluating trajectories") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)
                pbar.update(1)
        
        return results


# ====== CLI Interface ======
async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent trajectories")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="trajectories/chat_histories",
        help="Directory containing trajectory JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent evaluations"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM service"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API URL for LLM service"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Update API config
    if args.api_key:
        API_CONFIG['api_key'] = args.api_key
    if args.api_url:
        API_CONFIG['base_url'] = args.api_url
    if args.model:
        API_CONFIG['model'] = args.model
    
    # Run evaluation pipeline
    pipeline = EvaluationPipeline(API_CONFIG, args.output_dir)
    results = await pipeline.process_directory(args.input_dir, args.max_concurrent)
    
    # Generate summary report
    summary_file = Path(args.output_dir) / "evaluation_summary.json"
    summary = {
        'total_trajectories': len(results),
        'average_score': sum(r['overall_score'] for r in results) / len(results) if results else 0,
        'success_rate': sum(1 for r in results if r['success']) / len(results) if results else 0,
        'trajectories': [
            {
                'task_id': r['task_id'],
                'score': r['overall_score'],
                'success': r['success']
            }
            for r in results
        ]
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {len(results)} trajectories evaluated")
    print(f"Average score: {summary['average_score']:.2f}/5")
    print(f"Success rate: {summary['success_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Enhanced Agent Trajectory Evaluation System with Failure Attribution (Single Stage Scoring)

Evaluates agent trajectories by scoring four cognitive modules with failure type classification:
- Memory Recall
- Reflection  
- Planning/Thinking
- Action Selection

Uses LLM-based evaluation with simplified single-stage scoring for clarity.
"""

import json
import os
import asyncio
import aiohttp
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FailurePatterns:
    """Track failure patterns across trajectory"""
    consecutive_failures: int = 0
    same_strategy_repeats: int = 0
    total_failures: int = 0
    failure_types_by_step: Dict[int, str] = None
    action_loop_count: int = 0
    
    def __post_init__(self):
        if self.failure_types_by_step is None:
            self.failure_types_by_step = {}


@dataclass
class ModuleEvaluation:
    """Individual module evaluation result"""
    score: int  # 0-5 integer
    failure_type: str
    reasoning: str
    evidence: str
    suggestions: str


@dataclass
class TrajectoryEvaluation:
    """Complete trajectory evaluation results with failure attribution"""
    task_id: str
    task_description: str
    success: bool
    trajectory_text: str
    module_evaluations: Dict[str, List[ModuleEvaluation]]  # module_name -> list of step evaluations
    overall_score: float
    failure_analysis: Dict[str, Any]  # Overall failure patterns and attribution
    critical_issues: List[str]
    recommendations: List[str]

class FailureTracker:
    """Track and analyze failure patterns across trajectory"""
    
    def __init__(self):
        self.failure_indicators = [
            r"nothing happens",
            r"failed to",
            r"cannot",
            r"unable to",
            r"error",
            r"unsuccessful"
        ]
        self.compiled_indicators = [re.compile(pattern, re.IGNORECASE) for pattern in self.failure_indicators]
    
    def track_trajectory_failures(self, trajectory_data: Dict[str, Any]) -> Dict[int, FailurePatterns]:
        """Track failure patterns for each step in trajectory"""
        chat_history = trajectory_data.get('chat_history', [])
        failure_by_step = {}
        
        consecutive_failures = 0
        same_strategy_count = 0
        last_action_type = None
        recent_actions = deque(maxlen=5)
        total_failures = 0
        
        step_num = 0
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                step_num += 1
                
                # Get next user message (environment response)
                env_response = ""
                if i + 1 < len(chat_history) and chat_history[i + 1]['role'] == 'user':
                    env_response = chat_history[i + 1]['content']
                
                # Check for failure indicators
                is_failure = any(pattern.search(env_response) for pattern in self.compiled_indicators)
                
                if is_failure:
                    consecutive_failures += 1
                    total_failures += 1
                else:
                    consecutive_failures = 0
                
                # Extract and categorize action
                current_action = self._extract_action_type(msg['content'])
                
                # Track strategy repetition
                if current_action == last_action_type:
                    same_strategy_count += 1
                else:
                    same_strategy_count = 0
                
                # Track action loops
                loop_count = sum(1 for action in recent_actions if action == current_action)
                recent_actions.append(current_action)
                
                failure_by_step[step_num] = FailurePatterns(
                    consecutive_failures=consecutive_failures,
                    same_strategy_repeats=same_strategy_count,
                    total_failures=total_failures,
                    action_loop_count=loop_count
                )
                
                last_action_type = current_action
        
        return failure_by_step
    
    def _extract_action_type(self, content: str) -> str:
        """Extract and categorize action type from agent response"""
        action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL | re.IGNORECASE)
        if not action_match:
            return "unknown"
        
        action_content = action_match.group(1).strip().lower()
        
        # Categorize action types
        if any(keyword in action_content for keyword in ['go to', 'move to', 'navigate']):
            return "navigation"
        elif any(keyword in action_content for keyword in ['take', 'pick up', 'grab']):
            return "manipulation"  
        elif any(keyword in action_content for keyword in ['look', 'examine', 'inspect']):
            return "observation"
        elif any(keyword in action_content for keyword in ['open', 'close']):
            return "interaction"
        else:
            return "other"

class TrajectoryEvaluator:
    """Enhanced evaluator with single-stage scoring"""
    
    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        self.headers = {
            "Authorization": f"Bearer {api_config['api_key']}",
            "Content-Type": "application/json"
        }
        self.failure_tracker = FailureTracker()
    
    def get_failure_types(self) -> Dict[str, List[str]]:
        """Define failure types for each module"""
        return {
            'memory_recall': [
                'no_failure',           # 5 points
                'minor_gaps',           # 4 points  
                'limited_learning',     # 3 points
                'repetitive_memory',    # 2 points
                'inappropriate_usage',  # 1 point
                'false_memory'          # 0 points
            ],
            'reflection': [
                'no_failure',           # 5 points
                'minor_gaps',           # 4 points
                'basic_reflection',     # 3 points
                'shallow_analysis',     # 2 points
                'misinterpret_outcomes', # 1 point
                'no_reflection'         # 0 points
            ],
            'thinking': [
                'no_failure',           # 5 points
                'minor_gaps',           # 4 points
                'basic_planning',       # 3 points
                'vague_plan',          # 2 points
                'wrong_strategy',       # 1 point
                'no_planning'          # 0 points
            ],
            'action': [
                'no_failure',           # 5 points
                'minor_suboptimal',     # 4 points
                'reasonable_choice',    # 3 points
                'poor_reasoning',       # 2 points
                'wrong_action',         # 1 point
                'invalid_action'        # 0 points
            ]
        }
    
    def get_enhanced_rubrics(self) -> Dict[str, str]:
        """Simplified scoring rubrics with integrated failure consideration"""
        return {
            'memory_recall': """
Memory Recall Module Scoring with Failure Attribution (0-5 integers):

5 - Excellent (no_failure):
- Perfectly recalls relevant experiences and learns from them
- Makes sophisticated connections between past and current situations
- Clear evidence of strategic learning from previous failures
- Adapts approach based on accumulated experience

4 - Good (minor_gaps):
- Recalls most relevant information with only minor omissions
- Generally learns from experience with occasional missed connections
- Shows awareness of past attempts with good utilization

3 - Adequate (limited_learning):
- Recalls basic information but limited evidence of learning
- Some connections made but inconsistent application
- Remembers facts but struggles to apply lessons strategically

2 - Poor (repetitive_memory):
- Repeats same information without adding strategic value
- Fails to learn from repeated failures of same approach
- Memory content doesn't inform better decision making

1 - Very Poor (inappropriate_usage):
- Memory appears when it shouldn't (e.g., first step) 
- Completely irrelevant memory content for current situation
- Memory actively misleads or confuses the decision process

0 - Failure (false_memory):
- Fabricated or contradictory memory content
- Remembers things that never happened in the trajectory
- Complete absence of memory when critically needed

IMPORTANT: Consider failure history when scoring. Repeated failed strategies (5+ times) should receive lower scores even if content seems well-written.
            """,
            
            'reflection': """
Reflection Module Scoring with Failure Attribution (0-5 integers):

5 - Excellent (no_failure):
- Deep analysis of outcomes with clear causal understanding
- Identifies specific reasons for both successes and failures  
- Generates actionable insights for strategy improvement
- Shows metacognitive awareness and self-correction

4 - Good (minor_gaps):
- Solid analysis with mostly accurate interpretation
- Recognizes key outcomes and draws useful conclusions
- Some insights generated for future improvement

3 - Adequate (basic_reflection):
- Basic recognition of obvious outcomes
- Limited causal analysis but acknowledges results
- Some attempt at learning from experience

2 - Poor (shallow_analysis):
- Superficial reflection that misses key implications
- Recognizes outcomes but fails to understand causes
- Little to no insight generated for improvement

1 - Very Poor (misinterpret_outcomes):
- Misunderstands clear outcomes or their significance
- Blames external factors inappropriately
- Reflection actively hinders learning

0 - Failure (no_reflection):
- Complete absence of reflection when needed
- Contradicts clearly observable outcomes
- No evidence of any analytical thinking

IMPORTANT: After 8+ failures without strategy change, shallow analysis that doesn't propose alternatives should receive lower scores.
            """,
            
            'thinking': """
Planning/Thinking Module Scoring with Failure Attribution (0-5 integers):

5 - Excellent (no_failure):
- Clear, logical step-by-step planning with strategic reasoning
- Considers multiple approaches and evaluates trade-offs
- Anticipates obstacles and has contingency plans
- Shows evidence of learning from previous planning failures

4 - Good (minor_gaps):
- Solid planning with mostly clear logical progression
- Good alignment with task objectives
- Some strategic considerations present

3 - Adequate (basic_planning):
- Basic planning present with reasonable structure
- Generally moves toward goals but limited strategic depth
- Addresses immediate needs adequately

2 - Poor (vague_plan):
- Unclear or inconsistent planning
- Limited logical reasoning connecting steps to goals
- Plans don't effectively advance toward completion

1 - Very Poor (wrong_strategy):
- Plans that work against task objectives
- Confused or contradictory strategic thinking
- Poor reasoning that will likely hinder progress

0 - Failure (no_planning):
- Complete absence of planning when needed
- Actively contradicts task requirements or established facts
- Incoherent or impossible plans

IMPORTANT: Plans that ignore demonstrated failures (10+ times) or repeat proven ineffective strategies should receive lower scores.
            """,
            
            'action': """
Action Module Scoring with Failure Attribution (0-5 integers):

5 - Excellent (no_failure):
- Optimal action choice given current context and constraints
- Perfect alignment with stated plan and task objectives
- Efficient progress toward completion
- Shows learning from previous action failures

4 - Good (minor_suboptimal):
- Good action that advances task effectively
- Well-aligned with plan and objectives
- Minor inefficiencies but solid progress

3 - Adequate (reasonable_choice):
- Reasonable action that makes some progress
- Generally aligned with goals and context
- Acceptable choice given the situation

2 - Poor (poor_reasoning):
- Suboptimal action with questionable reasoning
- Poor alignment with plan or limited effectiveness
- May create additional problems to solve

1 - Very Poor (wrong_action):
- Wrong action that hinders progress toward goals
- Misaligned with plan and poor judgment shown
- Actively works against task completion

0 - Failure (invalid_action):
- Completely invalid action that violates constraints
- Impossible or nonsensical action choice
- Dangerous or harmful action selection

IMPORTANT: Actions that repeat the same failed approach (5+ times) should receive lower scores regardless of surface appropriateness.
            """
        }
    
    def get_enhanced_examples(self) -> Dict[str, str]:
        """Enhanced examples with failure type classification"""
        return {
            'memory_recall': """
ENHANCED EXAMPLES WITH FAILURE TYPES:

Example 1 - Score: 5, Failure Type: no_failure
Context: Multi-step task after several failed attempts
Content: "I previously tried opening cabinets 1-3 but found them empty. Cabinet 4 contained dishes but no mugs. I learned that kitchen items seem organized by type, so I should check areas specifically for drinkware like the coffee station I noticed earlier."
Reasoning: Excellent recall with strategic learning and adaptation.

Example 2 - Score: 2, Failure Type: repetitive_memory  
Context: After 8 failed attempts at same strategy
Content: "I remember I need to find a clean mug and have been checking cabinets."
Reasoning: Repeats obvious information without learning from failures.

Example 3 - Score: 0, Failure Type: false_memory
Context: First attempt at task
Content: "Based on my previous experience in this kitchen..."
Reasoning: Claims non-existent memory in first step.

Example 4 - Score: 1, Failure Type: inappropriate_usage
Context: After multiple "Nothing happens" responses
Content: "I remember that drawers usually contain utensils."
Reasoning: Generic memory that doesn't address why current strategy keeps failing.
            """,
            
            'reflection': """
ENHANCED EXAMPLES WITH FAILURE TYPES:

Example 1 - Score: 5, Failure Type: no_failure
Context: After failed attempt to use key
Content: "The key didn't work because I can see it's a car key, not a house key. This explains the failure - wrong tool entirely. I need to search for the correct house key in typical storage locations."
Reasoning: Identifies specific cause and provides clear next steps.

Example 2 - Score: 2, Failure Type: shallow_analysis
Context: After 15 failed "go to" actions
Content: "The actions didn't work. I should try other locations."
Reasoning: Acknowledges failure but doesn't analyze why or propose different approach.

Example 3 - Score: 0, Failure Type: no_reflection
Context: After clear failure
Content: "I will continue with the next step."
Reasoning: No reflection on outcome when it's clearly needed.

Example 4 - Score: 1, Failure Type: misinterpret_outcomes
Context: Successfully completed subtask
Content: "This failed because the environment is broken."
Reasoning: Misinterprets successful outcome as failure.
            """,
            
            'thinking': """
ENHANCED EXAMPLES WITH FAILURE TYPES:

Example 1 - Score: 5, Failure Type: no_failure
Context: Complex assembly task
Content: "I need to assemble this systematically. First organize pieces, then gather tools from the toolbox I saw. Start with base frame for stability. If pieces are missing, check packaging before proceeding."
Reasoning: Clear systematic approach with contingency planning.

Example 2 - Score: 2, Failure Type: vague_plan
Context: Search task
Content: "I should look around and find it somehow."
Reasoning: No specific plan or systematic approach.

Example 3 - Score: 0, Failure Type: no_planning
Context: Task requiring multi-step approach
Content: [No thinking content provided]
Reasoning: Complete absence of planning when needed.

Example 4 - Score: 1, Failure Type: wrong_strategy
Context: After multiple failures, task requires finding items
Content: "I'll just try random actions until something works."
Reasoning: Plan actively works against systematic task completion.
            """,
            
            'action': """
ENHANCED EXAMPLES WITH FAILURE TYPES:

Example 1 - Score: 5, Failure Type: no_failure
Context: Plan to check kitchen cabinet for utensils
Content: "action_choice: 'go to kitchen cabinet 1'"
Reasoning: Perfect alignment with plan and logical for objective.

Example 2 - Score: 2, Failure Type: poor_reasoning
Context: Looking for clean items in garbage
Content: "action_choice: 'go to garbage can'"
Reasoning: Poor reasoning - unlikely location for clean items.

Example 3 - Score: 0, Failure Type: invalid_action
Context: Standard environment
Content: "action_choice: 'teleport to cabinet'"
Reasoning: Impossible action not available in environment.

Example 4 - Score: 1, Failure Type: wrong_action
Context: After 15+ failures of same action type
Content: "action_choice: 'go to cabinet 1'"
Reasoning: Continues same failing action without adaptation.
            """
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
        
        # Track failure patterns
        failure_patterns = self.failure_tracker.track_trajectory_failures(data)
        
        return {
            'task_id': metadata.get('timestamp', 'unknown'),
            'task_description': task_description,
            'success': metadata.get('success', False),
            'trajectory_text': json.dumps(chat_history, indent=2),
            'modules': modules,
            'metadata': metadata,
            'failure_patterns': failure_patterns  # 添加这一行
        }
    
    async def evaluate_module(
        self,
        module_name: str,
        module_instances: List[Dict[str, Any]],
        trajectory_data: Dict[str, Any]
    ) -> List[ModuleEvaluation]:
        """Evaluate module with failure attribution"""
        
        rubrics = self.get_enhanced_rubrics()
        examples = self.get_enhanced_examples()
        failure_patterns = trajectory_data.get('failure_patterns', {})
        
        evaluations = []
        
        for instance in module_instances:
            step_num = instance['step']
            content = instance['content']
            
            # Get failure context for this step
            failure_context = failure_patterns.get(step_num, FailurePatterns())
            
            prompt = self._build_enhanced_prompt(
                module_name, content, step_num, trajectory_data, 
                failure_context, rubrics, examples
            )
            
            response = await self.call_llm(prompt)
            evaluation = self._parse_enhanced_response(response, failure_context)
            
            evaluations.append(evaluation)
        
        return evaluations
    
    def _build_enhanced_prompt(
        self,
        module_name: str,
        content: str,
        step_num: int,
        trajectory_data: Dict[str, Any],
        failure_context: FailurePatterns,
        rubrics: Dict[str, str],
        examples: Dict[str, str]
    ) -> str:
        """Build simplified evaluation prompt with integrated failure consideration"""
        
        failure_info = f"""
FAILURE CONTEXT FOR STEP {step_num}:
- Consecutive failures: {failure_context.consecutive_failures}
- Same strategy repeats: {failure_context.same_strategy_repeats} 
- Total failures so far: {failure_context.total_failures}
- Action loop count: {failure_context.action_loop_count}

IMPORTANT: Consider this failure history when scoring. Repeated failed strategies should receive lower scores regardless of surface quality.
"""
        
        prompt = f"""
You are an expert evaluator assessing agent performance with STRICT failure attribution.

TASK: {trajectory_data['task_description']}
TASK SUCCESS: {trajectory_data['success']}

{failure_info}

EVALUATION RUBRIC:
{rubrics[module_name]}

EXAMPLES:
{examples[module_name]}

CONTENT TO EVALUATE:
Step {step_num} - {module_name.replace('_', ' ').title()}: {content}

EVALUATION INSTRUCTIONS:
1. Consider BOTH content quality AND failure history when scoring
2. Repeated failed strategies should get lower scores even if well-written
3. Assign ONE integer score (0-5) that reflects overall performance
4. Choose the failure type that best describes the primary issue

REQUIRED OUTPUT FORMAT:
<reasoning>
Detailed analysis considering both content quality and failure context. Explain your final score.
</reasoning>

<score>
[0-5 integer considering both content and failure patterns]
</score>

<failure_type>
[exact failure type from rubric list]
</failure_type>

<evidence>
Specific evidence supporting the score and failure type.
</evidence>

<suggestions>
Concrete suggestions for improvement.
</suggestions>
"""
        return prompt
    
    def _parse_enhanced_response(self, response: str, failure_context: FailurePatterns) -> ModuleEvaluation:
        """Parse simplified evaluation response"""
        
        # Extract components with regex
        reasoning = self._extract_field(response, 'reasoning', 'No reasoning provided')
        score = int(self._extract_field(response, 'score', '0'))
        failure_type = self._extract_field(response, 'failure_type', 'unknown')
        evidence = self._extract_field(response, 'evidence', 'No evidence provided')
        suggestions = self._extract_field(response, 'suggestions', 'No suggestions provided')
        
        # Ensure score is in valid range
        score = max(0, min(5, score))
        
        return ModuleEvaluation(
            score=score,
            failure_type=failure_type,
            reasoning=reasoning,
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _extract_field(self, response: str, field_name: str, default: str = "") -> str:
        """Extract field from response"""
        pattern = f'<{field_name}>(.*?)</{field_name}>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else default
    
    async def call_llm(self, prompt: str) -> str:
        """Make API call to LLM"""
        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "system", "content": "You are an expert evaluator of agent performance with strict failure attribution standards. Always provide integer scores 0-5 and exact failure types."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config['temperature']
        }

        # 读取系统代理设置
        proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy') or os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        if proxy:
            print(f"检测到代理: {proxy}")

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config['max_retries']):
                try:
                    async with session.post(
                        self.config['base_url'],
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config['timeout']),
                        proxy=proxy if proxy else None
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
    
    # 继续实现其余方法...
    async def evaluate_trajectory(self, trajectory_data: Dict[str, Any]) -> TrajectoryEvaluation:
        """Evaluate complete trajectory with enhanced failure analysis"""
        
        module_evaluations = {}
        all_scores = []
        
        # Evaluate each module
        for module_name, module_instances in trajectory_data['modules'].items():
            if module_instances:
                evaluations = await self.evaluate_module(
                    module_name,
                    module_instances,
                    trajectory_data
                )
                module_evaluations[module_name] = evaluations
                all_scores.extend([eval.score for eval in evaluations])
        
        # Calculate overall score
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Analyze failures across trajectory
        failure_analysis = self._analyze_trajectory_failures(module_evaluations, trajectory_data)
        
        # Generate critical issues and recommendations
        critical_issues, recommendations = self._generate_insights(module_evaluations, failure_analysis)
        
        return TrajectoryEvaluation(
            task_id=trajectory_data['task_id'],
            task_description=trajectory_data['task_description'],
            success=trajectory_data['success'],
            trajectory_text=trajectory_data['trajectory_text'],
            module_evaluations=module_evaluations,
            overall_score=overall_score,
            failure_analysis=failure_analysis,
            critical_issues=critical_issues,
            recommendations=recommendations
        )
    
    def _analyze_trajectory_failures(
        self, 
        module_evaluations: Dict[str, List[ModuleEvaluation]], 
        trajectory_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze failure patterns across the entire trajectory"""
        
        failure_counts = defaultdict(int)
        score_trends = defaultdict(list)
        critical_steps = []
        
        # Collect failure statistics
        for module_name, evaluations in module_evaluations.items():
            for eval in evaluations:
                failure_counts[eval.failure_type] += 1
                score_trends[module_name].append(eval.score)
                
                if eval.score <= 1:
                    critical_steps.append({
                        'module': module_name,
                        'score': eval.score,
                        'failure_type': eval.failure_type,
                        'reasoning': eval.reasoning[:100] + "..."
                    })
        
        # Calculate trends
        trend_analysis = {}
        for module_name, scores in score_trends.items():
            if len(scores) > 2:
                early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
                late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
                trend_analysis[module_name] = {
                    'early_performance': early_avg,
                    'late_performance': late_avg,
                    'trend': 'improving' if late_avg > early_avg else 'declining' if late_avg < early_avg else 'stable'
                }
        
        return {
            'failure_type_distribution': dict(failure_counts),
            'performance_trends': trend_analysis,
            'critical_steps': critical_steps[:5],  # Top 5 worst steps
            'total_critical_failures': len([s for s in critical_steps if s['score'] <= 1]),
            'pattern_summary': self._summarize_patterns(failure_counts, trend_analysis)
        }
    
    def _summarize_patterns(self, failure_counts: Dict, trends: Dict) -> str:
        """Generate human-readable pattern summary"""
        top_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        summary = "Key patterns: "
        if top_failures:
            summary += f"Most common failure: {top_failures[0][0]} ({top_failures[0][1]} instances). "
        
        declining_modules = [name for name, data in trends.items() if data['trend'] == 'declining']
        if declining_modules:
            summary += f"Declining performance in: {', '.join(declining_modules)}. "
        
        improving_modules = [name for name, data in trends.items() if data['trend'] == 'improving']
        if improving_modules:
            summary += f"Improving performance in: {', '.join(improving_modules)}."
        
        return summary
    
    def _generate_insights(
        self, 
        module_evaluations: Dict[str, List[ModuleEvaluation]], 
        failure_analysis: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Generate critical issues and recommendations"""
        
        critical_issues = []
        recommendations = []
        
        # Check for systemic issues
        if failure_analysis['total_critical_failures'] > 3:
            critical_issues.append(f"Systemic failure: {failure_analysis['total_critical_failures']} critical failures detected")
        
        # Module-specific issues
        for module_name, evaluations in module_evaluations.items():
            avg_score = sum(eval.score for eval in evaluations) / len(evaluations)
            if avg_score <= 2:
                critical_issues.append(f"{module_name.replace('_', ' ').title()} consistently poor (avg: {avg_score:.1f})")
                recommendations.append(f"Focus on improving {module_name.replace('_', ' ')} - see module-specific suggestions")
        
        # Pattern-based recommendations
        failure_dist = failure_analysis['failure_type_distribution']
        if failure_dist.get('repetitive_memory', 0) > 2:
            recommendations.append("Implement explicit failure tracking in memory module")
        
        if failure_dist.get('shallow_analysis', 0) > 2:
            recommendations.append("Enhance reflection with deeper causal analysis requirements")
        
        return critical_issues, recommendations

    async def process_file(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Process a single trajectory file with enhanced evaluation"""
        try:
            # Parse trajectory
            trajectory_data = self.parse_chat_history(file_path)
            
            # Evaluate trajectory
            evaluation = await self.evaluate_trajectory(trajectory_data)
            
            # Generate training data files for each module evaluation
            base_filename = Path(file_path).stem
            
            for module_name, module_evaluations in evaluation.module_evaluations.items():
                for i, eval_result in enumerate(module_evaluations):
                    # Create training data format
                    training_data = {
                        "instruction": self._build_training_instruction(module_name),
                        "input": self._build_training_input(
                            module_name, 
                            trajectory_data['modules'][module_name][i], 
                            trajectory_data,
                            i + 1  # step number
                        ),
                        "output": self._build_training_output(eval_result)
                    }
                    
                    # Save module-specific file
                    module_file = Path(output_dir) / f"{base_filename}_{module_name}_step{i+1}.json"
                    with open(module_file, 'w', encoding='utf-8') as f:
                        json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            # Save comprehensive evaluation report
            report_file = Path(output_dir) / f"{base_filename}_evaluation_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "task_id": evaluation.task_id,
                    "task_description": evaluation.task_description,
                    "success": evaluation.success,
                    "overall_score": evaluation.overall_score,
                    "failure_analysis": evaluation.failure_analysis,
                    "critical_issues": evaluation.critical_issues,
                    "recommendations": evaluation.recommendations,
                    "module_summaries": self._summarize_module_performance(evaluation.module_evaluations)
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluated {evaluation.task_id}: Score {evaluation.overall_score:.2f}")
            
            return {
                'task_id': evaluation.task_id,
                'task_description': evaluation.task_description,
                'success': evaluation.success,
                'overall_score': evaluation.overall_score,
                'failure_analysis': evaluation.failure_analysis
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _build_training_instruction(self, module_name: str) -> str:
        """Build instruction for training data"""
        rubrics = self.get_enhanced_rubrics()
        examples = self.get_enhanced_examples()
        
        return f"""
You are an expert evaluator assessing agent performance in autonomous task execution.
Evaluate the {module_name.replace('_', ' ').title()} module with strict failure attribution.

EVALUATION RUBRIC:
{rubrics[module_name]}

EXAMPLES:
{examples[module_name]}

EVALUATION INSTRUCTIONS:
1. Consider BOTH content quality AND failure history when scoring
2. Repeated failed strategies should get lower scores even if well-written
3. Assign ONE integer score (0-5) that reflects overall performance
4. Choose the failure type that best describes the primary issue
5. Provide specific evidence and improvement suggestions

REQUIRED OUTPUT FORMAT:
<reasoning>Detailed analysis considering both content quality and failure context. Explain your final score.</reasoning>
<score>[0-5 integer considering both content and failure patterns]</score>
<failure_type>[exact failure type from rubric list]</failure_type>
<evidence>Specific evidence supporting the score and failure type.</evidence>
<suggestions>Concrete suggestions for improvement.</suggestions>
        """.strip()
    

    def _build_training_input(
        self, 
        module_name: str, 
        module_instance: Dict[str, Any], 
        trajectory_data: Dict[str, Any],
        step_num: int
    ) -> str:
        """Build input for training data - chat_history format"""
        
        failure_context = trajectory_data.get('failure_patterns', {}).get(step_num, FailurePatterns())
        
        # Get step context from chat_history
        trajectory_text = trajectory_data.get('trajectory_text', '')
        chat_history = json.loads(trajectory_text) if trajectory_text else []
        
        # Find current step context from chat_history
        current_step_context = ""
        assistant_count = 0
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                assistant_count += 1
                if assistant_count == step_num:
                    # Get the user message before this assistant message (environment response)
                    env_response = ""
                    if i > 0 and chat_history[i-1]['role'] == 'user':
                        env_response = chat_history[i-1]['content'][:200]
                    
                    # Get the assistant's action from this message
                    action_match = re.search(r'<action>(.*?)</action>', msg['content'], re.DOTALL)
                    action = action_match.group(1).strip() if action_match else "No action found"
                    
                    current_step_context = f"Step {step_num} - Environment: {env_response}... Agent Action: {action}"
                    break
        
        return f"""
    TASK CONTEXT:
    Task: {trajectory_data['task_description']}
    Task Success: {trajectory_data['success']}
    Current Step: {step_num}

    STEP CONTEXT:
    {current_step_context}

    FAILURE CONTEXT:
    - Consecutive failures: {failure_context.consecutive_failures}
    - Same strategy repeats: {failure_context.same_strategy_repeats}
    - Total failures so far: {failure_context.total_failures}
    - Action loop count: {failure_context.action_loop_count}

    MODULE CONTENT TO EVALUATE:
    {module_name.replace('_', ' ').title()}: {module_instance['content']}

    RECENT TRAJECTORY HISTORY:
    {self._get_recent_chat_context(chat_history, step_num, 3)}
        """.strip()

        
    def _get_recent_chat_context(self, chat_history: List[Dict], current_step: int, window: int = 3) -> str:
        """Get recent steps context from chat_history for training data"""
        context_steps = []
        start_step = max(1, current_step - window)
        
        assistant_count = 0
        for i, msg in enumerate(chat_history):
            if msg['role'] == 'assistant':
                assistant_count += 1
                
                # Only include steps within the window
                if start_step <= assistant_count <= current_step:
                    # Get environment response (previous user message)
                    env_response = ""
                    if i > 0 and chat_history[i-1]['role'] == 'user':
                        env_response = chat_history[i-1]['content'][:100]
                    
                    # Get agent action
                    action_match = re.search(r'<action>(.*?)</action>', msg['content'], re.DOTALL)
                    action = action_match.group(1).strip()[:50] if action_match else "No action"
                    
                    context_steps.append(f"Step {assistant_count}: {env_response}... → {action}")
        
        return "\n".join(context_steps)
    
    def _build_training_output(self, eval_result: ModuleEvaluation) -> str:
        """Build simplified training output from evaluation result"""
        
        return f"""
<reasoning>
{eval_result.reasoning}
</reasoning>

<score>
{eval_result.score}
</score>

<failure_type>
{eval_result.failure_type}
</failure_type>

<evidence>
{eval_result.evidence}
</evidence>

<suggestions>
{eval_result.suggestions}
</suggestions>
        """.strip()
    
    def _summarize_module_performance(self, module_evaluations: Dict[str, List[ModuleEvaluation]]) -> Dict[str, Any]:
        """Summarize performance for each module"""
        summaries = {}
        
        for module_name, evaluations in module_evaluations.items():
            if not evaluations:
                continue
                
            scores = [eval.score for eval in evaluations]
            failure_types = [eval.failure_type for eval in evaluations]
            
            # Calculate statistics
            avg_score = sum(scores) / len(scores)
            failure_distribution = {}
            for ft in failure_types:
                failure_distribution[ft] = failure_distribution.get(ft, 0) + 1
            
            # Identify trends
            if len(scores) > 2:
                early_scores = scores[:len(scores)//2]
                late_scores = scores[len(scores)//2:]
                early_avg = sum(early_scores) / len(early_scores)
                late_avg = sum(late_scores) / len(late_scores)
                trend = 'improving' if late_avg > early_avg + 0.5 else 'declining' if late_avg < early_avg - 0.5 else 'stable'
            else:
                trend = 'insufficient_data'
            
            summaries[module_name] = {
                'average_score': round(avg_score, 2),
                'score_range': f"{min(scores)}-{max(scores)}",
                'total_evaluations': len(evaluations),
                'failure_distribution': failure_distribution,
                'performance_trend': trend,
                'most_common_failure': max(failure_distribution.items(), key=lambda x: x[1])[0] if failure_distribution else 'none',
                'critical_failures': len([s for s in scores if s <= 1])
            }
        
        return summaries
    
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
        with tqdm(total=len(tasks), desc="Evaluating trajectories with failure attribution") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result:
                    results.append(result)
                pbar.update(1)
        
        # Generate comprehensive summary
        self.generate_enhanced_summary(results, output_dir)
        
        return results
    
    def generate_enhanced_summary(self, results: List[Dict[str, Any]], output_dir: str):
        """Generate enhanced evaluation summary with failure analysis"""
        
        if not results:
            return
        
        from datetime import datetime
        
        # Calculate overall statistics
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        scores = [r['overall_score'] for r in results]
        
        # Failure analysis aggregation
        all_failure_types = defaultdict(int)
        critical_patterns = []
        
        for result in results:
            failure_analysis = result.get('failure_analysis', {})
            failure_dist = failure_analysis.get('failure_type_distribution', {})
            
            for failure_type, count in failure_dist.items():
                all_failure_types[failure_type] += count
            
            if failure_analysis.get('total_critical_failures', 0) > 3:
                critical_patterns.append({
                    'task_id': result['task_id'],
                    'critical_failures': failure_analysis['total_critical_failures'],
                    'pattern': failure_analysis.get('pattern_summary', 'Unknown pattern')
                })
        
        # Score distribution
        score_distribution = {
            'excellent (4-5)': sum(1 for s in scores if s >= 4),
            'good (3-4)': sum(1 for s in scores if 3 <= s < 4),
            'poor (1-3)': sum(1 for s in scores if 1 <= s < 3),
            'failure (0-1)': sum(1 for s in scores if s < 1)
        }
        
        # Create enhanced summary
        summary = {
            'evaluation_timestamp': str(datetime.now()),
            'evaluation_type': 'enhanced_with_failure_attribution_single_stage',
            'overview': {
                'total_trajectories': total,
                'successful_trajectories': successful,
                'success_rate': round(successful / total, 3),
                'average_overall_score': round(sum(scores) / len(scores), 3),
                'score_distribution': score_distribution
            },
            'failure_analysis': {
                'most_common_failures': dict(sorted(all_failure_types.items(), key=lambda x: x[1], reverse=True)[:10]),
                'trajectories_with_critical_patterns': len(critical_patterns),
                'critical_pattern_examples': critical_patterns[:5]
            },
            'performance_insights': {
                'high_performers': sorted(results, key=lambda x: x['overall_score'], reverse=True)[:3],
                'needs_attention': sorted([r for r in results if r['overall_score'] < 2], key=lambda x: x['overall_score'])[:5]
            },
            'recommendations': self._generate_global_recommendations(all_failure_types, score_distribution)
        }
        
        # Save enhanced summary
        summary_file = Path(output_dir) / "enhanced_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print enhanced summary
        print(f"\n{'='*70}")
        print("ENHANCED TRAJECTORY EVALUATION SUMMARY (Single Stage Scoring)")
        print(f"{'='*70}")
        print(f"Total Trajectories: {total}")
        print(f"Success Rate: {successful}/{total} ({successful/total:.1%})")
        print(f"Average Score: {sum(scores)/len(scores):.2f}/5.0")
        print(f"\nScore Distribution:")
        for category, count in score_distribution.items():
            print(f"  {category}: {count} ({count/total:.1%})")
        print(f"\nTop Failure Types:")
        for failure_type, count in list(all_failure_types.items())[:5]:
            print(f"  {failure_type}: {count} instances")
        print(f"\nCritical Pattern Trajectories: {len(critical_patterns)}")
        print(f"\nResults saved to: {output_dir}")
        print(f"Enhanced summary: {summary_file}")
    
    def _generate_global_recommendations(self, failure_types: Dict, score_distribution: Dict) -> List[str]:
        """Generate global recommendations based on failure patterns"""
        recommendations = []
        
        total_evaluations = sum(failure_types.values())
        if total_evaluations == 0:
            return ["Insufficient data for recommendations"]
        
        # Check for systemic issues
        if failure_types.get('repetitive_memory', 0) / total_evaluations > 0.2:
            recommendations.append("High repetitive memory failures - implement explicit failure tracking systems")
        
        if failure_types.get('shallow_analysis', 0) / total_evaluations > 0.2:
            recommendations.append("Frequent shallow analysis - enhance reflection requirements with causal reasoning")
        
        if failure_types.get('wrong_strategy', 0) / total_evaluations > 0.15:
            recommendations.append("Strategy selection issues - improve planning with failure history consideration")
        
        if failure_types.get('wrong_action', 0) / total_evaluations > 0.15:
            recommendations.append("Action selection problems - implement action validity checking and learning")
        
        # Score-based recommendations
        if score_distribution.get('failure (0-1)', 0) > len(score_distribution) * 0.3:
            recommendations.append("High failure rate - focus on basic competency training before advanced skills")
        
        if score_distribution.get('excellent (4-5)', 0) < len(score_distribution) * 0.2:
            recommendations.append("Few excellent performances - investigate top performers for best practices")
        
        return recommendations


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Evaluate agent trajectories with failure attribution (single stage)")
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
    asyncio.run(main())
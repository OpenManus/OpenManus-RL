"""
LLM Judge Reward Manager for Tool Use Environment
Evaluates agent answers against ground truth using LLM judge similar to calculate_score.py
"""

import re
import json
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel
from collections import defaultdict

try:
    from openmanus_rl.engines.openai import ChatOpenAI
    HAS_LLM_ENGINE = True
except ImportError:
    HAS_LLM_ENGINE = False
    logging.warning("LLM engine not available. LLM judge reward will not work.")


class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool


class LLMJudgeRewardManager:
    """
    Manages LLM-based reward calculation for tool use tasks.
    Evaluates agent answers against ground truth using GPT-based judging.
    """
    
    def __init__(self, model_string: str = "gpt-4o-mini"):
        self.model_string = model_string
        self.llm_engine = None
        if HAS_LLM_ENGINE:
            try:
                self.llm_engine = ChatOpenAI(
                    model_string=model_string,
                    is_multimodal=False,
                    enable_cache=True
                )
                print(f"LLM Judge initialized with {model_string}")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM engine: {e}")
    
    def extract_final_answers(self, responses: List[str]) -> List[str]:
        """
        Extract final answers from agent responses.
        Looks for <answer>...</answer> tags or FINAL_ANSWER: format.
        """
        final_answers = []
        
        for response in responses:
            answer = ""
            
            # Try to find <answer> tags first
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            # Fallback to FINAL_ANSWER format
            elif "FINAL_ANSWER:" in response:
                lines = response.split('\n')
                for line in lines:
                    if line.strip().startswith("FINAL_ANSWER:"):
                        answer = line.split("FINAL_ANSWER:", 1)[1].strip()
                        break
            
            # If no structured answer found, try to extract from end of response
            if not answer:
                # Look for common answer patterns at the end
                lines = response.split('\n')
                for line in reversed(lines[-5:]):  # Check last 5 lines
                    line = line.strip()
                    if line and not line.startswith(('<', 'Tool', 'Result:')):
                        answer = line
                        break
            
            final_answers.append(answer)
        
        return final_answers
    
    def evaluate_answers(self, agent_answers: List[str], ground_truths: List[str], 
                        pids: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate agent answers against ground truth using LLM judge.
        
        Returns:
            Dict containing:
            - individual_scores: List of 0/1 scores for each answer
            - individual_analyses: List of analysis texts
            - overall_accuracy: Float accuracy score
        """
        if not self.llm_engine:
            print("Warning: LLM engine not available, returning zero scores")
            return {
                'individual_scores': [0.0] * len(agent_answers),
                'individual_analyses': ['LLM judge not available'] * len(agent_answers),
                'overall_accuracy': 0.0
            }
        
        individual_scores = []
        individual_analyses = []
        
        for i, (agent_answer, ground_truth) in enumerate(zip(agent_answers, ground_truths)):
            if not agent_answer.strip():
                # Empty answer - automatically wrong
                individual_scores.append(0.0)
                individual_analyses.append("Empty answer provided")
                continue
            
            try:
                analysis, is_correct = self._judge_single_answer(agent_answer, ground_truth)
                individual_scores.append(1.0 if is_correct else 0.0)
                individual_analyses.append(analysis)
            except Exception as e:
                print(f"Error evaluating answer {i}: {e}")
                individual_scores.append(0.0)
                individual_analyses.append(f"Evaluation error: {str(e)}")
        
        overall_accuracy = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        
        return {
            'individual_scores': individual_scores,
            'individual_analyses': individual_analyses,
            'overall_accuracy': overall_accuracy,
            'correct_count': sum(individual_scores),
            'total_count': len(individual_scores)
        }
    
    def _judge_single_answer(self, agent_answer: str, ground_truth: str) -> tuple:
        """
        Judge a single answer against ground truth using LLM.
        Returns (analysis, is_correct).
        """
        query_prompt = f"""
        Compare the model's response against the correct answer following these evaluation rules:

        Model response: {agent_answer}
        Correct answer: {ground_truth}

        Evaluation rules:
        1. Extract the core answer from the model response (ignore explanations or additional context)
        2. The answer is correct if it EXACTLY matches the correct answer:
           - Numbers must match precisely (e.g., "142" = "142")
           - Text must match case-sensitive (e.g., "Time-Parking 2: Parallel Universe")
           - Zip codes must be exact (e.g., "34689")
           - No partial credit for similar or related answers
        3. The answer is incorrect if:
           - It contains any additional or missing information
           - It uses different formatting or representations
           - It's semantically equivalent but not identical

        Response Format:
        <analysis>: Extract the core answer and explain exact match comparison
        <true_false>: Return "True" only for exact matches, otherwise "False"
        """

        verification = self.llm_engine(query_prompt, response_format=AnswerVerification)
        
        analysis = verification.analysis.strip()
        is_correct = verification.true_false
        
        return analysis, is_correct


def calculate_delayed_rewards(episode_data: List[Dict], reward_manager: LLMJudgeRewardManager) -> Dict[str, Any]:
    """
    Calculate delayed rewards for completed episodes using LLM judge.
    
    Args:
        episode_data: List of episode dictionaries containing responses and ground truth
        reward_manager: LLMJudgeRewardManager instance
        
    Returns:
        Dict containing reward scores and metadata
    """
    # Extract final answers and ground truths
    responses = [ep.get('final_response', '') for ep in episode_data]
    ground_truths = [ep.get('ground_truth', '') for ep in episode_data]
    pids = [ep.get('pid', str(i)) for i, ep in enumerate(episode_data)]
    
    # Extract final answers from responses
    final_answers = reward_manager.extract_final_answers(responses)
    
    # Evaluate using LLM judge
    evaluation_results = reward_manager.evaluate_answers(final_answers, ground_truths, pids)
    
    # Convert to reward format
    rewards = evaluation_results['individual_scores']
    
    return {
        'rewards': rewards,
        'final_answers': final_answers,
        'analyses': evaluation_results['individual_analyses'],
        'accuracy': evaluation_results['overall_accuracy'],
        'correct_count': evaluation_results['correct_count'],
        'total_count': evaluation_results['total_count']
    }

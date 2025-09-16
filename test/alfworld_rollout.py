#!/usr/bin/env python3
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests

# Configure project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / '.env')

from openmanus_rl.multi_turn_rollout.openmanus_rollout import OpenmanusRollout
from openmanus_rl.environments.env_manager import make_envs
from openmanus_rl.environments.prompts.alfworld import ALFWORLD_OPENMANUS_INITIAL_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration with sensible defaults."""
    batch_size: int = 1
    max_steps: int = 10
    seed: int = 42
    save_trajectories: bool = True
    output_dir: str = "trajectories"
    history_window: int = 3
    
    @property
    def env_config(self):
        return {
            'env_name': 'alfworld/AlfredTWEnv',
            'seed': self.seed,
            'max_steps': self.max_steps,
            'history_length': self.history_window,
            'rollout': type('RolloutConfig', (), {'n': 0})()
        }


class TrajectoryStep:
    """Single step in a trajectory with full state information."""
    
    def __init__(self, step_num: int):
        self.step = step_num
        self.observation_before = None
        self.admissible_actions = []
        self.llm_prompt = None
        self.llm_response = None
        self.parsed_action = None
        self.reward = 0.0
        self.done = False
        self.won = False
        self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step': self.step,
            'state': {
                'observation': self.observation_before,
                'admissible_actions': self.admissible_actions
            },
            'agent_output': {
                'raw_response': self.llm_response,
                'action': self.parsed_action
            },
            'transition': {
                'reward': self.reward,
                'done': self.done
            },
            'metadata': self.metadata
        }


class LLMAgent:
    """Agent that interfaces with LLM APIs using chat-based conversation."""
    
    def __init__(self):
        # Check environment for API credentials
        self._setup_api()
        self.chat_history = []  # Store chat messages for conversation flow
        self.current_task = None
        self.is_first_turn = True
        
    def _setup_api(self):
        """Configure API based on environment variables."""
        # Support both OpenAI and Azure OpenAI
        self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OAI_KEY')
        self.api_base = os.getenv('OPENAI_API_BASE') or os.getenv('OAI_ENDPOINT')
        self.api_type = os.getenv('OPENAI_API_TYPE', 'openai')  # 'openai' or 'azure'
        
        if self.api_key:
            self.api_enabled = True
            if self.api_type == 'azure' and self.api_base:
                logger.info(f"Azure OpenAI configured: {self.api_base[:30]}...")
            elif self.api_type == 'openai':
                if not self.api_base:
                    self.api_base = 'https://api.openai.com'
                logger.info(f"OpenAI API configured")
            else:
                self.api_enabled = False
                logger.warning("Invalid API configuration")
        else:
            self.api_enabled = False
            logger.warning("No API credentials found, using heuristic fallback")
    
    def reset(self, task_description: str):
        """Reset agent state for new episode."""
        self.chat_history = []
        self.current_task = task_description
        self.is_first_turn = True
        
        # Add system message to initialize the conversation
        self.chat_history.append({
            "role": "system", 
            "content": "You are an expert AI agent solving household tasks in the ALFRED environment."
        })
    
    def act(self, observation: str, admissible_actions: List[str]) -> Tuple[str, str]:
        """
        Generate action based on current observation using chat conversation.
        
        Returns:
            Tuple of (raw_response, action)
        """
        if self.is_first_turn:
            # First turn: use initial template with task description
            user_message = self._create_initial_prompt(observation, admissible_actions)
            self.is_first_turn = False
        else:
            # Subsequent turns: just provide observation and actions
            user_message = self._create_followup_message(observation, admissible_actions)
        
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Get response from LLM only, no fallback
        if not self.api_enabled:
            raise RuntimeError("API not configured. Please set OPENAI_API_KEY and OPENAI_API_BASE")
        
        response = self._query_llm_chat()
        
        # Add assistant response to chat history
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Keep chat history bounded (keep system message + last 10 exchanges)
        if len(self.chat_history) > 21:  # 1 system + 20 user/assistant messages
            # Keep system message and last 10 exchanges (20 messages)
            self.chat_history = [self.chat_history[0]] + self.chat_history[-20:]
        
        return response, self._extract_action(response)
    
    def _create_initial_prompt(self, observation: str, actions: List[str]) -> str:
        """Create initial prompt using the template for first turn."""
        return ALFWORLD_OPENMANUS_INITIAL_TEMPLATE.format(
            task_description=self.current_task or "Complete the task",
            current_observation=observation,
            admissible_actions=", ".join(actions) if actions else "none available"
        )
    
    def _create_followup_message(self, observation: str, actions: List[str]) -> str:
        """Create followup message for subsequent turns."""
        return f"Observation: {observation}\n\nAvailable actions: [{', '.join(actions) if actions else 'none available'}]\n\nPlease respond with your memory recall, reflection, thinking, and action as instructed."
    
    def _query_llm_chat(self) -> str:
        """Query the LLM API using chat history."""
        try:
            # Set headers based on API type
            if self.api_type == 'azure':
                headers = {
                    "api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                url = f"{self.api_base}/openai/deployments/gpt-4o/chat/completions?api-version=2024-05-13"
            else:  # OpenAI API
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                url = f"{self.api_base}/v1/chat/completions"
            
            payload = {
                "model": "gpt-4o" if self.api_type == 'openai' else None,  # OpenAI needs model in payload
                "messages": self.chat_history,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # Remove None values from payload
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                logger.debug(f"LLM response received: {len(content)} chars")
                
                # Check if response was truncated (missing action tags)
                if '<think>' in content and not ('</action>' in content or '</reflect' in content.lower()):
                    logger.warning(f"Response appears truncated, missing action tags")
                    # Could implement retry with simpler prompt here
                
                return content
            else:
                logger.error(f"API error {response.status_code}: {response.text[:200]}")
                raise RuntimeError(f"API request failed with status {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            logger.error(f"API exception: {e}")
            raise RuntimeError(f"API request failed: {e}")
    
    
    def _extract_action(self, response: str) -> str:
        """Extract action from structured response."""
        if '<action>' in response and '</action>' in response:
            start = response.find('<action>') + 8
            end = response.find('</action>')
            action_text = response[start:end].strip()
            
            # Handle different action formats
            if 'action_choice:' in action_text:
                parts = action_text.split('action_choice:')
                if len(parts) > 1:
                    action = parts[1].split('\n')[0].strip()
                    # Remove quotes if present
                    action = action.strip("'\"")
                    return action
            
            # Return first line if no special format, removing quotes
            action = action_text.split('\n')[0].strip()
            action = action.strip("'\"")
            return action
        
        # Smarter fallback: try to extract meaningful action from response
        response_lower = response.lower()
        
        # Look for common action patterns in the thinking
        if 'go to cabinet' in response_lower:
            # Extract cabinet number
            import re
            match = re.search(r'go to cabinet (\d+)', response_lower)
            if match:
                return f"go to cabinet {match.group(1)}"
        
        if 'open cabinet' in response_lower:
            match = re.search(r'open cabinet (\d+)', response_lower)
            if match:
                return f"open cabinet {match.group(1)}"
                
        if 'go to drawer' in response_lower:
            match = re.search(r'go to drawer (\d+)', response_lower)
            if match:
                return f"go to drawer {match.group(1)}"
        
        # Default fallback
        return "look"


class TrajectoryCollector:
    """Manages trajectory collection and storage."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.trajectories = []
        self._setup_output_dir()
    
    def _setup_output_dir(self):
        """Create output directory if needed."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def collect(self, env, agent, rollout_processor) -> Dict[str, Any]:
        """
        Collect a single trajectory.
        
        Returns:
            Dictionary containing the full trajectory data.
        """
        trajectory = []
        obs, _ = env.reset()
        
        # Initialize agent with task
        task_description = obs['text'][0]
        agent.reset(task_description)
        
        logger.info(f"Starting trajectory collection for task: {task_description[:100]}...")
        
        for step_num in range(self.config.max_steps):
            # Create step record
            step = TrajectoryStep(step_num + 1)
            step.observation_before = obs['text'][0]
            step.admissible_actions = obs.get('admissible_actions', [None])[0] or []
            
            # Generate action
            raw_response, _ = agent.act(step.observation_before, step.admissible_actions)
            step.llm_response = raw_response
            
            # Process response through rollout system
            action, _ = rollout_processor.process_response(
                raw_response,
                episode_id=f"ep_{datetime.now().strftime('%H%M%S')}",
                step_id=step_num
            )
            step.parsed_action = action or "look"
            
            # Validate action before execution
            if step.admissible_actions and step.parsed_action not in step.admissible_actions:
                logger.warning(f"Invalid action '{step.parsed_action}', using 'look' instead")
                step.parsed_action = "look"
            
            # Execute in environment
            next_obs, rewards, dones, infos = env.step([step.parsed_action])
            
            step.reward = float(rewards[0])
            step.done = bool(dones[0])
            # Convert any numpy arrays in info to lists for JSON serialization
            info_dict = infos[0] if infos else {}
            
            # Extract admissible actions from info if not already set
            if not step.admissible_actions and 'admissible_commands' in info_dict:
                step.admissible_actions = info_dict['admissible_commands']
            
            # Store metadata excluding admissible_commands (to avoid duplication)
            step.metadata = {
                'info': {k: v.tolist() if hasattr(v, 'tolist') else v 
                        for k, v in info_dict.items() 
                        if k != 'admissible_commands'}
            }
            
            # Store won status for success determination
            step.won = info_dict.get('won', False)
            
            trajectory.append(step)
            
            # Check termination - success or environment done
            if step.done:
                logger.info(f"Episode completed at step {step_num + 1}")
                break
            elif step.won:
                logger.info(f"Task completed successfully at step {step_num + 1}!")
                break
            
            obs = next_obs
        
        return {
            'task': task_description,
            'steps': [s.to_dict() for s in trajectory],
            'total_reward': sum(s.reward for s in trajectory),
            'success': any(s.won for s in trajectory),  # True if any step shows won=True
            'length': len(trajectory)
        }
    
    def save(self, trajectory: Dict[str, Any], run_id: str = None) -> str:
        """Save trajectory to JSON file."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = Path(self.config.output_dir) / f"traj_{run_id}.json"
        
        # Add metadata
        output = {
            'metadata': {
                'timestamp': run_id,
                'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
                'environment': 'alfworld',
                'version': '1.0'
            },
            'trajectory': trajectory
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        file_size_kb = os.path.getsize(filename) / 1024
        logger.info(f"Saved trajectory to {filename} ({file_size_kb:.2f} KB)")
        
        return str(filename)


def run_experiment(config: Optional[ExperimentConfig] = None) -> bool:
    """
    Run a complete trajectory collection experiment.
    
    Args:
        config: Experiment configuration (uses defaults if None)
    
    Returns:
        Success status
    """
    if config is None:
        config = ExperimentConfig()
    
    logger.info("Starting AlfWorld trajectory collection")
    logger.info(f"Configuration: batch_size={config.batch_size}, max_steps={config.max_steps}")
    
    try:
        # Initialize environment
        logger.info("Initializing environment...")
        
        # Create minimal config for environment
        env_config = type('Config', (), {
            'env': type('EnvConfig', (), config.env_config)(),
            'data': type('DataConfig', (), {
                'train_batch_size': config.batch_size,
                'val_batch_size': 1
            })()
        })()
        
        envs, _ = make_envs(env_config)
        
        # Initialize components
        agent = LLMAgent()
        collector = TrajectoryCollector(config)
        
        # Simple tokenizer stub
        tokenizer = type('Tokenizer', (), {'pad_token_id': 0})()
        rollout = OpenmanusRollout(env_config, tokenizer, None)
        
        # Collect trajectory
        trajectory = collector.collect(envs, agent, rollout)
        
        # Save results
        if config.save_trajectories:
            saved_path = collector.save(trajectory)
            logger.info(f"Experiment complete. Results saved to {saved_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAJECTORY COLLECTION SUMMARY")
        print("="*50)
        print(f"Task: {trajectory['task'][:80]}...")
        print(f"Steps taken: {trajectory['length']}")
        print(f"Total reward: {trajectory['total_reward']:.2f}")
        print(f"Success: {'Yes' if trajectory['success'] else 'No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            envs.close()
        except:
            pass


if __name__ == "__main__":
    # Parse command line args if needed
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect AlfWorld trajectories')
    parser.add_argument('--steps', type=int, default=10, help='Max steps per episode')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--num_tasks', type=int, default=1, help='Number of tasks to run')
    parser.add_argument('--no-save', action='store_true', help='Disable trajectory saving')
    
    args = parser.parse_args()
    
    # Configure experiment
    exp_config = ExperimentConfig(
        max_steps=args.steps,
        batch_size=args.batch,
        save_trajectories=not args.no_save
    )
    
    # Run multiple tasks if requested
    successes = 0
    for task_idx in range(args.num_tasks):
        logger.info(f"\n=== Running task {task_idx + 1}/{args.num_tasks} ===")
        if run_experiment(exp_config):
            successes += 1
    
    logger.info(f"\n=== Completed {successes}/{args.num_tasks} tasks successfully ===")
    sys.exit(0 if successes == args.num_tasks else 1)
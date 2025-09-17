from typing import List, Tuple, Dict, Union, Any
import torch
import numpy as np
import os
from openmanus_rl.environments.prompts import *
from collections import defaultdict

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, (int, float, bool, Tuple, List)):
        data = np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)})")
    return data

class EnvironmentManagerBase:
    def __init__(self, envs, projection_f, config):
        """
        Initialize the environment manager.
        
        Parameters:
        - envs: The environment instance, usually a vectorized environment containing multiple sub-environments.
        - projection_f: A function that maps text actions to environment actions.
        - config: Configuration object.
        """
        self.envs = envs
        self.projection_f = projection_f
        self.config = config
        
        # Debugger-related attributes
        self.debugger_feedback = {}  # {env_id: {'step': int, 'feedback': str}}
        self.replay_mode = {}  # {env_id: bool} - whether in replay mode
        self.replay_actions = {}  # {env_id: List[str]} - actions to replay
        self.current_replay_step = {}  # {env_id: int} - current step in replay
        self.persistent_guidance = {}  # {env_id: {'start_step': int, 'text': str}}

    def reset(self) -> Dict[str, Any]:
        """
        Reset all environments and return the initial observations.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        """
        obs, infos = self.envs.reset()
        return {'text': None, 'image': obs, 'anchor': None}, infos
    
    def step(self, text_actions: List[str]):
        """
        Execute text actions and return the next state, rewards, done flags, and additional information.
        
        Parameters:
        - text_actions (List[str]): A list of text actions to execute.
        
        Returns:
        - next_observations (Dict):
          - 'text' (None or List[str]): The textual observation.
          - 'image' (np.ndarray or torch.Tensor): The image observation as either a NumPy array or a PyTorch tensor.
          - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        - rewards (np.ndarry or torch.Tensor): The rewards returned by the environment.
        - dones (np.ndarray or torch.Tensor): Done flags indicating which environments have completed.
        - infos (List[Dict]): Additional environment information.
        
        Exceptions:
        - NotImplementedError: If an observation key is not in ('text', 'image').
        """
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_observations = {
            'text': None, # Implement this if needed
            'image': next_obs,
            'anchor': None # For GiGPO only. anchor observation without any histories, hint, etc. Implement this if needed
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        
        return next_observations, rewards, dones, infos

    def build_text_obs(self,) -> List[str]:
        """
        This function builds the text observation for the agent.
        
        Returns:
        - postprocess_text_obs (List[str]): A list of processed text observations.
        """
        pass
    
    def setup_replay(
        self,
        env_id: int,
        actions_to_replay: List[str],
        debugger_feedback_step: int,
        debugger_feedback_text: str,
        persistent_guidance_text: str | None = None,
        persistent_guidance_start: int | None = None,
    ):
        """
        Setup replay mode for a specific environment.
        
        Parameters:
        - env_id: The environment ID to setup replay for
        - actions_to_replay: List of actions to replay up to the critical step
        - debugger_feedback_step: The step where debugger feedback should be injected (0-based)
        - debugger_feedback_text: The feedback text to inject
        """
        import logging
        logging.info(f"    setup_replay called: env_id={env_id}, feedback_step={debugger_feedback_step}, actions={len(actions_to_replay)}")
        
        self.replay_mode[env_id] = True
        self.replay_actions[env_id] = actions_to_replay
        self.current_replay_step[env_id] = 0
        self.debugger_feedback[env_id] = {
            'step': debugger_feedback_step,
            'feedback': debugger_feedback_text
        }
        
        # Configure persistent guidance (used by continue-mode debugger)
        if persistent_guidance_text:
            start_step = persistent_guidance_start if persistent_guidance_start is not None else debugger_feedback_step
            self.set_persistent_guidance(env_id, persistent_guidance_text, start_step)
        else:
            self.clear_persistent_guidance(env_id)

        logging.info(f"    setup_replay complete: debugger_feedback keys = {list(self.debugger_feedback.keys())}")
        
    def is_in_replay_mode(self, env_id: int) -> bool:
        """Check if environment is in replay mode."""
        return self.replay_mode.get(env_id, False)

    def set_persistent_guidance(self, env_id: int, guidance_text: str, start_step: int = 0):
        """Register guidance that should persist on all observations from start_step onwards."""
        self.persistent_guidance[env_id] = {
            'start_step': max(0, int(start_step)),
            'text': guidance_text,
        }

    def get_persistent_guidance(self, env_id: int, current_step: int) -> str:
        """Return persistent guidance text if current_step is past the configured start."""
        guidance = self.persistent_guidance.get(env_id)
        if not guidance:
            return ""

        if current_step >= guidance.get('start_step', 0):
            return guidance.get('text', "")

        return ""

    def clear_persistent_guidance(self, env_id: int):
        """Remove persistent guidance for the specified environment."""
        self.persistent_guidance.pop(env_id, None)
    
    def get_replay_action(self, env_id: int) -> str:
        """Get the current replay action and advance replay step."""
        if not self.is_in_replay_mode(env_id):
            return None
            
        replay_step = self.current_replay_step.get(env_id, 0)
        actions = self.replay_actions.get(env_id, [])
        
        if replay_step < len(actions):
            action = actions[replay_step]
            self.current_replay_step[env_id] = replay_step + 1
            return action
        else:
            # No more replay actions – keep debugger feedback available for injection
            return None
    
    def clear_replay(self, env_id: int):
        """Clear replay mode for a specific environment."""
        self.replay_mode.pop(env_id, None)
        self.replay_actions.pop(env_id, None)
        self.current_replay_step.pop(env_id, None)
        self.debugger_feedback.pop(env_id, None)
    
    def get_debugger_feedback(self, env_id: int, current_step: int) -> str:
        """
        Get debugger feedback if it should be injected at the current step.
        
        Parameters:
        - env_id: The environment ID
        - current_step: The current step number
        
        Returns:
        - feedback: The feedback string or empty string if no feedback for this step
        """
        import logging
        logging.info(f"    get_debugger_feedback called: env_id={env_id}, current_step={current_step}")
        
        if env_id in self.debugger_feedback:
            feedback_data = self.debugger_feedback[env_id]
            expected_step = feedback_data['step']
            logging.info(f"    debugger_feedback exists: expected_step={expected_step}, current_step={current_step}")
            
            if feedback_data['step'] == current_step:
                logging.info(f"    ✓ Injecting debugger feedback at env_id={env_id}, step={current_step}")
                feedback_text = feedback_data['feedback']
                # Replay sequence complete – clear stored state so it doesn't leak into future attempts
                self.clear_replay(env_id)
                return feedback_text
            else:
                logging.info(f"    × Step mismatch: expected={expected_step}, current={current_step}")
        else:
            logging.info(f"    × No debugger_feedback for env_id={env_id}")
            
        return ""

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.envs.close()

    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                return
            
    def save_image(self, image, step):
        """
        Save an image to a file.
        
        Parameters:
        - image (np.ndarray or torch.Tensor): The image to save.
        - path (str): The path to save the image.
        """
        path = os.path.join(os.path.dirname(__file__), os.path.join("images", self.config.env.env_name))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f"step{step}.png")
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type: {type(image)})")
        
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        if image.max() <= 1.0:
            image = (image * 255)

        image = image.astype(np.uint8)
        
        from PIL import Image
        image = Image.fromarray(image)
        image.save(path)

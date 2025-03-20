from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Any, Iterator
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy

# define text objects

@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool

TextHistory = Tuple[Text, ...]
text_history_to_str = lambda text_history: ''.join(map(lambda x: x.text, text_history))
StepResult = Tuple[TextHistory, float, bool]

# text trajectory should fit into a single context window, otherwise is truncated

@dataclass(frozen=True)
class TextTrajectory:
    text_history: TextHistory
    reward: Tuple[float, ...]
    done: bool

    def __post_init__(self):
        assert len(self.reward) == len(self.text_history), "reward is needed for each text"
        assert all([r == 0.0 for r, t in zip(self.reward, self.text_history) if not t.is_action]), "reward for non-actions texts should be 0.0"

# text trajectory chain is a linked list of text trajectories
@dataclass(frozen=True)
class TextTrajectoryChain:
    text_trajectory: TextTrajectory
    next: Optional[TextTrajectoryChain]

# text environment

class TextEnv(ABC):
    @abstractmethod
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None) -> TextHistory:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> TextEnv:
        return deepcopy(self)

class BatchedTextEnv(ABC):
    @abstractmethod
    def step(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[Tuple[TextHistory, float, bool]]]:
        pass

    @abstractmethod
    def reset(self, seed: Optional[List[Optional[int]]]=None, options: Optional[List[Optional[Dict]]]=None) -> List[TextHistory]:
        pass

    def close(self) -> None:
        pass

    def copy(self) -> BatchedTextEnv:
        return deepcopy(self)

class TextEnvToBatchedTextEnv(BatchedTextEnv):
    def __init__(self, env: TextEnv):
        self.env = env
        self.batch_env_copies = None

    def step(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[Tuple[TextHistory, float, bool]]]:
        assert self.batch_env_copies is not None, 'reset must be called before step'
        assert len(text_history) == len(self.batch_env_copies), 'batch size must be the same as the number of environments initalized'
        if done is None:
            done = [False]*len(text_history)
        assert len(text_history) == len(done)
        return [None if d else env.step(item) for env, item, d in zip(self.batch_env_copies, text_history, done)]
    
    def reset(self, seed: Optional[List[Optional[int]]]=None, options: Optional[List[Optional[Dict]]]=None) -> List[TextHistory]:
        if seed is None and options is None:
            seed, options = [None], [None]
        elif seed is None:
            seed = [None] * len(options)
        elif options is None:
            options = [None] * len(seed)
        assert len(seed) == len(options)
        self.batch_env_copies = [self.env.copy() for _ in range(len(seed))]
        return [env.reset(seed=s, options=o) for env, s, o in zip(self.batch_env_copies, seed, options)]
    
    def close(self) -> None:
        for env in self.batch_env_copies:
            env.close()
        self.env.close()

class BatchedTextEnvToTextEnv(TextEnv):
    def __init__(self, env: BatchedTextEnv):
        self.env = env

    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        return self.env.step([text_history])[0]
    
    def reset(self, seed: Optional[int]=None, options: Optional[Dict]=None) -> TextHistory:
        return self.env.reset(seed=[seed], options=[options])[0]
    
    def close(self) -> None:
        self.env.close()

# text policy

class TextPolicy(ABC):
    @abstractmethod
    def act(self, text_history: TextHistory) -> TextHistory:
        pass

class BatchedTextPolicy(ABC):
    @abstractmethod
    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[TextHistory]]:
        pass

class TextPolicyToBatchedTextPolicy(BatchedTextPolicy):
    def __init__(self, policy: TextPolicy):
        self.policy = policy
    
    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[TextHistory]]:
        print(done)
        print(text_history)
        print(len(text_history))
        if done is None:
            done = [False]*len(text_history)
        assert len(text_history) == len(done)
        return [None if d else self.policy.act(item) for item, d in zip(text_history, done)]

class BatchedTextPolicyToTextPolicy(TextPolicy):
    def __init__(self, policy: BatchedTextPolicy):
        self.policy = policy
    
    def act(self, text_history: TextHistory) -> TextHistory:
        return self.policy.act([text_history])[0]

# interact with the environment

class InteractionTransition(NamedTuple):
    pre_action_history: TextHistory
    post_action_history: TextHistory
    post_transition_history: TextHistory
    reward: float
    done: bool

def interact_environment(
    env: Union[TextEnv, BatchedTextEnv], 
    policy: Union[TextPolicy, BatchedTextPolicy], 
    initial_text_history: Optional[Union[TextHistory, List[TextHistory]]]=None, 
    env_seed: Union[Optional[int], Optional[List[Optional[int]]]]=None, 
    env_options: Union[Optional[Dict], Optional[List[Optional[int]]]]=None, 
    bsize: int=1, 
    npad: int=0,
) -> List[List[InteractionTransition]]:
    assert bsize > 0
    if isinstance(env, TextEnv):
        env = TextEnvToBatchedTextEnv(env)
    if isinstance(policy, TextPolicy):
        policy = TextPolicyToBatchedTextPolicy(policy)
    if env_seed is not None and isinstance(env_seed, int):
        env_seed = [env_seed] * bsize
    if env_options is not None and isinstance(env_options, dict):
        env_options = [env_options] * bsize
    if initial_text_history is not None and isinstance(initial_text_history, TextHistory):
        initial_text_history = [initial_text_history] * bsize
    text_history = initial_text_history
    if text_history is None:
        text_history = env.reset(env_seed, env_options)
    
    transitions_batch = [[] for _ in range(bsize)]
    done = [False]*bsize
    while not all(done):
        pre_action_history = text_history
        text_history = policy.act(text_history + [(Text("", is_action=False),)]*npad, done=done + [True]*npad)
        text_history = text_history[:bsize]
        post_action_history = text_history

        step_results = env.step(text_history, done=done)
        step_results = list(map(lambda x: (None, None, True) if x is None else x, step_results))
        text_history, reward, done = (list(x) for x in zip(*step_results))
        post_transition_history = text_history
        
        for batch_idx in range(bsize):
            if done[batch_idx] and \
                (pre_action_history[batch_idx] is None or \
                 post_action_history[batch_idx] is None or \
                 post_transition_history[batch_idx] is None or \
                 reward[batch_idx] is None):
                continue
            transitions_batch[batch_idx].append(
                InteractionTransition(
                    pre_action_history=pre_action_history[batch_idx], 
                    post_action_history=post_action_history[batch_idx], 
                    post_transition_history=post_transition_history[batch_idx], 
                    reward=reward[batch_idx], 
                    done=done[batch_idx], 
                )
            )
    return transitions_batch

    

def text_env_eval(
    env: Union[TextEnv, BatchedTextEnv], 
    policy: Union[TextPolicy, BatchedTextPolicy], 
    n_rollouts: int, 
    initial_text_history: Optional[TextHistory]=None, # only allow one initial_text_history here
    seed_generator: Optional[Iterator[int]]=None, 
    env_options: Optional[Dict]=None, # only allow one env_options here
    interaction_callback: Optional[Callable[[List[Tuple[TextHistory, TextHistory, TextHistory, float, bool]]], None]]=None, 
    bsize: int=1, 
    verbose: bool=True, 
) -> Tuple[List[List[InteractionTransition]], Dict[str, Any]]:
    interactions, rewards, dones, eps_lengths = [], [], [], []
    for _ in tqdm(range((n_rollouts+(bsize-1))//bsize), disable=not verbose):
        actual_bsize = min(n_rollouts-len(interactions), bsize)
        npad = bsize - actual_bsize
        interaction_batch = interact_environment(
            env, 
            policy, 
            initial_text_history=initial_text_history, 
            env_seed=[None]*actual_bsize if seed_generator is None else [next(seed_generator) for _ in range(actual_bsize)], 
            env_options=[env_options]*actual_bsize, 
            bsize=actual_bsize,
            npad=npad,
        )
        
        for interaction in interaction_batch:
            interactions.append(interaction)
            rewards.append(sum(map(lambda x: x.reward, interaction)))
            dones.append(interaction[-1].done)
            eps_lengths.append(len(interaction))
            if interaction_callback is not None:
                interaction_callback(interaction)
    
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    results_summary = dict(
        reward=dict(
            mean=np.mean(rewards), 
            std=np.std(rewards), 
            min=np.min(rewards), 
            max=np.max(rewards), 
        ), 
        done=dict(
            mean=np.mean(dones), 
            std=np.std(dones), 
            min=np.min(dones), 
            max=np.max(dones), 
        ), 
        length=dict(
            mean=np.mean(eps_lengths),
            std=np.std(eps_lengths),
            min=np.min(eps_lengths),
            max=np.max(eps_lengths),
        ),
    )
    
    return interactions, results_summary

# user policy

class UserPolicy(TextPolicy):    
    def __init__(
        self, 
        initial_str: str, 
        postproc_print_f: Optional[Callable[[str], str]]=None, 
        postproc_action_f: Optional[Callable[[str], str]]=None, 
    ):
        self.initial_str = initial_str
        self.postproc_print_f = postproc_print_f if postproc_print_f is not None else lambda x: x
        self.postproc_action_f = postproc_action_f if postproc_action_f is not None else lambda x: x

    def act(self, text_history: TextHistory) -> TextHistory:
        print('='*25)
        print(self.postproc_print_f(text_history_to_str(text_history)))
        print('='*25)
        response = input(self.initial_str)
        response = self.initial_str + response
        return text_history+[Text(self.postproc_action_f(response), True)]


"""tokenize environment objects"""


@dataclass(frozen=True)
class TokenHistory:
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array

    def __post_init__(self):
        assert len(self.tokens.shape) == 1 and len(self.is_action.shape) == 1, '(tokens, is_action) must be 1 dimensional'
        assert self.tokens.shape == self.is_action.shape, '(tokens, is_action) must have the same shape'
    
    @classmethod
    def from_text_history(
        cls, 
        text_history: TextHistory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenHistory:
        if token_process is None:
            token_process = lambda x: x

        tokens = []
        is_action = []
        
        for item in text_history:
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))
        
        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
        )

@dataclass(frozen=True)
class TokenTrajectory:
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array
    reward: np.ndarray # 1d float32 array
    done: np.ndarray # bool scalar

    def __post_init__(self):
        assert len(self.tokens.shape) == 1, 'tokens must be 1 dimensional'
        assert len(self.is_action.shape) == 1, 'is_action must be 1 dimensional'
        assert len(self.reward.shape) == 1, 'reward must be 1 dimensional'
        assert len(self.done.shape) == 0, 'done must be scalar'

        assert self.is_action.shape == self.tokens.shape, 'is_action must have the same shape as tokens'
        assert self.reward.shape == self.tokens.shape, 'reward must have the same shape as tokens'

        assert not np.any(((1 - self.is_action.astype(np.float32)) * self.reward) != 0.0), 'reward must be 0.0 if not an action'
    
    @classmethod
    def from_text_trajectory(
        cls, 
        text_trajectory: TextTrajectory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectory:
        if token_process is None:
            token_process = lambda x: x
        
        tokens = []
        is_action = []
        reward = []

        for i, item in enumerate(text_trajectory.text_history):
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))
            
            # add reward at the last token in the text
            reward.extend(([0.0]*(len(new_tokens)-1))+[text_trajectory.reward[i]])
        
        # get done
        done = text_trajectory.done

        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
            np.array(reward, dtype=np.float32), 
            np.array(done, dtype=np.bool_), 
        )

@dataclass(frozen=True)
class TokenTrajectoryChain:
    token_trajectory: TokenTrajectory
    next: Optional[TokenTrajectoryChain]

    def __post_init__(self):
        curr, dones = self, []
        while curr.next is not None:
            dones.append(curr.token_trajectory.done)
            curr = curr.next
        assert not np.any(dones[:-1]), 'token trajectory chain can only be done at the end'
    
    def to_list(self) -> List[TokenTrajectory]:
        curr, l = self, []
        while curr is not None:
            l.append(curr.token_trajectory)
            curr = curr.next
        return l

    @classmethod
    def from_text_trajectory_chain(
        cls, 
        text_trajectory_chain: TextTrajectoryChain, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectoryChain:
        return TokenTrajectoryChain(
            TokenTrajectory.from_text_trajectory(
                text_trajectory_chain.text_trajectory, 
                tokenizer, 
                token_process=token_process, 
            ), 
            cls.from_text_trajectory_chain(
                text_trajectory_chain.next, 
                tokenizer, 
                token_process=token_process, 
            ) if text_trajectory_chain.next is not None else None, 
        )

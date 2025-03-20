from __future__ import annotations
from typing import Dict, Iterable, List, Iterator, NamedTuple
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy
import numpy as np
import jax.numpy as jnp
import jax
from transformers.tokenization_utils import PreTrainedTokenizerBase
from LLM_RL.environment import TokenTrajectoryChain

def get_rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    gamma_row = jnp.cumprod(jnp.full((rewards.shape[0],), gamma, dtype=jnp.float32), axis=0)
    gamma_tensor = jnp.triu(jnp.expand_dims(gamma_row, 0) / jnp.expand_dims(gamma_row, 1))
    reward2go = (gamma_tensor * jnp.expand_dims(rewards, 0)).sum(axis=1)
    return reward2go

class MCData(NamedTuple):
    input_ids: np.ndarray # [t]
    should_take_action: np.ndarray # [t-1]
    returns: np.ndarray # [t-1]

    @staticmethod
    def block(
        data: List[MCData], 
        blocking_strategy: BlockingStrategy, 
        tokenizer: PreTrainedTokenizerBase, 
    ) -> Dict[str, np.ndarray]:
        return dict(
            input_ids=block_sequences(
                list(map(lambda x: x.input_ids, data)), 
                tokenizer.pad_token_id, 
                dtype=np.int32, 
                blocking_strategy=blocking_strategy, 
            ), 
            should_take_action=block_sequences(
                list(map(lambda x: x.should_take_action, data)), 
                False, 
                dtype=np.bool_, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length-1), 
            ), 
            returns=block_sequences(
                list(map(lambda x: x.returns, data)), 
                0.0, 
                dtype=np.float32, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length-1), 
            ), 
        )
    
    @classmethod
    def from_token_trajectory_chain(
        cls, 
        token_trajectory_chain: TokenTrajectoryChain, 
        gamma: float, 
    ):
        filtered_rewards_chain = []
        should_take_action_chain = []
        for token_trajectory in token_trajectory_chain.to_list():
            should_take_action = token_trajectory.is_action[1:]
            rewards = token_trajectory.reward[1:]
            filtered_rewards = rewards[should_take_action]
            filtered_rewards_chain.append(filtered_rewards)
            should_take_action_chain.append(should_take_action)
        filtered_rewards_chain = np.concatenate(filtered_rewards_chain, axis=0)
        should_take_action_chain = np.concatenate(should_take_action_chain, axis=0)
        
        rtgs_sequence = get_rtg(filtered_rewards_chain, gamma=gamma)
        
        should_take_action = token_trajectory_chain.token_trajectory.is_action[1:]
        returns = np.zeros_like(should_take_action, dtype=np.float32)
        returns[should_take_action] = rtgs_sequence[:should_take_action.sum()]
        return cls(
            input_ids=token_trajectory_chain.token_trajectory.tokens, 
            should_take_action=should_take_action, 
            returns=returns, 
        )

class MCDataset(Dataset):
    def __init__(
        self, 
        input_ids: np.ndarray, # [b, t]
        should_take_action: np.ndarray, # [b, t-1]
        returns: np.ndarray, # [b, t-1]
    ):
        assert input_ids.shape[1] == (should_take_action.shape[1]+1)
        assert input_ids.shape[1] == (returns.shape[1]+1)

        assert input_ids.shape[0] == should_take_action.shape[0]
        assert input_ids.shape[0] == returns.shape[0]

        self.input_ids = input_ids
        self.should_take_action = should_take_action
        self.returns = returns
    
    def __getitem__(self, index):
        return {
            'input_ids': jnp.asarray(self.input_ids[index], dtype=jnp.int32), 
            'should_take_action': jnp.asarray(self.should_take_action[index], dtype=jnp.bool_), 
            'returns': jnp.asarray(self.returns[index], dtype=jnp.float32), 
        }
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    @classmethod
    def from_mc_data_list(
        cls, 
        mc_data_list: List[MCData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> MCDataset:
        
        data = MCData.block(mc_data_list, blocking_strategy, tokenizer)

        return cls(**data)

class _MCIteratorDataset:
    def __init__(self, mc_data: Iterator[Dict[str, np.ndarray]]):
        self.mc_data = mc_data

    def __next__(self):
        item = next(self.mc_data)
        return {
            'input_ids': jnp.asarray(item['input_ids'], dtype=jnp.int32), 
            'should_take_action': jnp.asarray(item['should_take_action'], dtype=jnp.bool_), 
            'returns': jnp.asarray(item['returns'], dtype=jnp.float32), 
        }

class MCIterableDataset(IterableDataset):
    def __init__(self, mc_data: Iterable[Dict[str, np.ndarray]]):
        self.mc_data = mc_data
    
    def __iter__(self):
        return _MCIteratorDataset(iter(self.mc_data))
    
    @classmethod
    def from_mc_data_iterable(
        cls, 
        mc_data: Iterable[MCData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> MCIterableDataset:
        
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for item in mc_data:
                    yield jax.tree_util.tree_map(lambda x: x[0], MCData.block([item], blocking_strategy, tokenizer))

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())

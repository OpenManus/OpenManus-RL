from __future__ import annotations
from typing import Dict, Iterable, List, Iterator, NamedTuple, Optional
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy
import numpy as np
import jax.numpy as jnp
import jax
from transformers.tokenization_utils import PreTrainedTokenizerBase
from LLM_RL.environment import TokenTrajectoryChain

class ILQLData(NamedTuple):
    input_ids: np.ndarray # [t]
    should_take_action: np.ndarray # [t-1]
    rewards: np.ndarray # [t-1]
    done: np.ndarray # []
    next_token_ids: Optional[np.ndarray] # [t']
    next_done: Optional[np.ndarray] # []

    @staticmethod
    def block(
        data: List[ILQLData], 
        blocking_strategy: BlockingStrategy, 
        tokenizer: PreTrainedTokenizerBase, 
    ) -> Dict[str, np.ndarray]:
        has_next_token = any(map(lambda x: x.next_token_ids is not None, data))
        assert all(map(lambda x: x.next_token_ids is None, data)) or has_next_token
        assert all(map(lambda x: x.next_done is None, data)) or has_next_token

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
            rewards=block_sequences(
                list(map(lambda x: x.rewards, data)), 
                0.0, 
                dtype=np.float32, 
                blocking_strategy=blocking_strategy._replace(max_length=blocking_strategy.max_length-1), 
            ), 
            dones=np.asarray(list(map(lambda x: x.done, data)), dtype=np.bool_), 
            next_token_ids=block_sequences(
                list(map(lambda x: x.next_token_ids, data)), 
                tokenizer.pad_token_id, 
                dtype=np.int32, 
                blocking_strategy=blocking_strategy, 
            ) if has_next_token else None, 
            next_dones=np.asarray(list(map(lambda x: x.next_done, data)), dtype=np.bool_) if has_next_token else None, 
        )
    
    @classmethod
    def from_token_trajectory_chain(
        cls, 
        token_trajectory_chain: TokenTrajectoryChain, 
    ):
        if token_trajectory_chain.next is not None:
            if token_trajectory_chain.next.token_trajectory.is_action[1:].sum() > 0:
                first_next_action = np.argmax(token_trajectory_chain.next.token_trajectory.is_action[1:], axis=0)+1
                next_token_ids = token_trajectory_chain.next.token_trajectory.tokens[:first_next_action]
                next_done = False
            else:
                next_token_ids = token_trajectory_chain.next.token_trajectory.tokens
                next_done = token_trajectory_chain.next.token_trajectory.done
        else:
            next_token_ids, next_done = None, None
        return cls(
            input_ids=token_trajectory_chain.token_trajectory.tokens, 
            should_take_action=token_trajectory_chain.token_trajectory.is_action[1:], 
            rewards=token_trajectory_chain.token_trajectory.reward[1:], 
            done=token_trajectory_chain.token_trajectory.done, 
            next_token_ids=next_token_ids, 
            next_done=next_done, 
        )

class ILQLDataset(Dataset):
    def __init__(
        self, 
        input_ids: np.ndarray, # [b, t]
        should_take_action: np.ndarray, # [b, t-1]
        rewards: np.ndarray, # [b, t-1]
        dones: np.ndarray, # [b]
        next_token_ids: Optional[np.ndarray], # [b, t']
        next_dones: Optional[np.ndarray], # [b]
    ):
        assert input_ids.shape[1] == (should_take_action.shape[1]+1)
        assert input_ids.shape[1] == (rewards.shape[1]+1)

        assert input_ids.shape[0] == should_take_action.shape[0]
        assert input_ids.shape[0] == rewards.shape[0]
        assert input_ids.shape[0] == dones.shape[0]
        if next_token_ids is not None:
            assert input_ids.shape[0] == next_token_ids.shape[0]
        if next_dones is not None:
            assert input_ids.shape[0] == next_dones.shape[0]

        self.input_ids = input_ids
        self.should_take_action = should_take_action
        self.rewards = rewards
        self.dones = dones
        self.next_token_ids = next_token_ids
        self.next_dones = next_dones
    
    def __getitem__(self, index):
        return {
            'input_ids': jnp.asarray(self.input_ids[index], dtype=jnp.int32), 
            'should_take_action': jnp.asarray(self.should_take_action[index], dtype=jnp.bool_), 
            'rewards': jnp.asarray(self.rewards[index], dtype=jnp.float32), 
            'dones': jnp.asarray(self.dones[index], dtype=jnp.float32), 
            'next_token_ids': jnp.asarray(self.next_token_ids[index], dtype=jnp.float32) if self.next_token_ids is not None else None, 
            'next_dones': jnp.asarray(self.next_dones[index], dtype=jnp.float32) if self.next_dones is not None else None, 
        }
    
    def __len__(self):
        return self.input_ids.shape[0]
    
    @classmethod
    def from_ilql_data_list(
        cls, 
        ilql_data_list: List[ILQLData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> ILQLDataset:
        
        data = ILQLData.block(ilql_data_list, blocking_strategy, tokenizer)

        return cls(**data)

class _ILQLIteratorDataset:
    def __init__(self, ilql_data: Iterator[Dict[str, np.ndarray]]):
        self.ilql_data = ilql_data

    def __next__(self):
        item = next(self.ilql_data)
        return {
            'input_ids': jnp.asarray(item['input_ids'], dtype=jnp.int32), 
            'should_take_action': jnp.asarray(item['should_take_action'], dtype=jnp.bool_), 
            'rewards': jnp.asarray(item['rewards'], dtype=jnp.float32), 
            'dones': jnp.asarray(item['dones'], dtype=jnp.float32), 
            'next_token_ids': jnp.asarray(item['next_token_ids'], dtype=jnp.float32) if item['next_token_ids'] is not None else None, 
            'next_dones': jnp.asarray(item['next_dones'], dtype=jnp.float32) if item['next_dones'] is not None else None, 
        }

class ILQLIterableDataset(IterableDataset):
    def __init__(self, ilql_data: Iterable[Dict[str, np.ndarray]]):
        self.ilql_data = ilql_data
    
    def __iter__(self):
        return _ILQLIteratorDataset(iter(self.ilql_data))
    
    @classmethod
    def from_ilql_data_iterable(
        cls, 
        ilql_data: Iterable[ILQLData], 
        tokenizer: PreTrainedTokenizerBase, 
        blocking_strategy: BlockingStrategy, 
    ) -> ILQLIterableDataset:
        
        class _TokensIterable(Iterable):
            def _tokens_generator(self):
                for item in ilql_data:
                    yield jax.tree_util.tree_map(lambda x: x[0], ILQLData.block([item], blocking_strategy, tokenizer))

            def __iter__(self):
                return self._tokens_generator()

        return cls(_TokensIterable())

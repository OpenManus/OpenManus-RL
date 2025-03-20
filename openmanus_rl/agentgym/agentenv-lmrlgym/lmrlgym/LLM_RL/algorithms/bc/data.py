from typing import Any, Callable, Generator, Iterator, List, Optional, Tuple, Union
from LLM_RL.environment import TokenHistory
import numpy as np
from  import Dataset, IterableDataset, block_sequences
import jax.numpy as jnp

# pad token histories

def block_token_histories(token_histories: List[TokenHistory], max_len: Optional[int], pad_token_id: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens = block_sequences(
        [token_history.tokens for token_history in token_histories], 
        max_len=max_len, 
        pad_value=pad_token_id, 
        dtype=np.int32, 
    )
    is_action = block_sequences(
        [token_history.is_action for token_history in token_histories], 
        max_len=max_len, 
        pad_value=False, 
        dtype=np.bool_, 
    )
    return tokens, is_action

# %BC filter data

def filter_items(score_fn: Callable[[Any], float], items: List[Any], 
                 take_top: Optional[float] = None, threshold: Optional[float] = None) -> List[Any]:
        assert ((take_top is None and threshold is not None)
                or (take_top is not None and threshold is None))

        scores = np.array([score_fn(item) for item in items])
        
        if take_top is not None:
            threshold = np.percentile(scores, 100 - take_top)

        new_items = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                new_items.append(items[i])
        
        return new_items

def filter_generator(score_fn: Callable[[Any], float], item_generator: Iterator[Any], threshold: float) -> Generator[float, None, None]:
    for item in item_generator:
        if score_fn(item) >= threshold:
            yield item

# datasets

class BCDataset(Dataset):
    def __init__(self, token_histories: List[TokenHistory], pad_token_id: int, max_len: Optional[int]):     
        self.tokens, self.is_action = block_token_histories(token_histories, max_len, pad_token_id)
    
    def __getitem__(self, idx):
        return jnp.asarray(self.tokens[idx], dtype=jnp.int32), jnp.asarray(self.is_action[idx], dtype=jnp.bool_)
    
    def __len__(self):
        return self.tokens.shape[0]

class BCIterableDataset(IterableDataset):
    def __init__(self, 
                 token_generator: Iterator[TokenHistory], 
                 pad_token_id: int, 
                 max_len: int):
        self.token_generator = token_generator
        self.pad_token_id = pad_token_id
        self.max_len = max_len

    def __iter__(self):
        return self
    
    def __next__(self):
        token_history = next(self.token_generator)
        
        tokens, is_action = block_token_histories([token_history], self.max_len, self.pad_token_id)
        tokens, is_action = tokens[0], is_action[0]
        
        return jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(is_action, dtype=jnp.bool_)
from LLM_RL.algorithms.ppo.base_interface import PPOInference
from JaxSeq.models.base_interface import Inference
from LLM_RL.environment import TextHistory, TokenHistory
from transformers.tokenization_utils import PreTrainedTokenizer
import jax.numpy as jnp
import numpy as np
import jax
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Any, Iterator

def build_ppo_score_fn(
    inference: PPOInference, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int, 
    bsize: int, 
):
    
    def score_fn(text_histories: List[TextHistory]) -> List[float]:
        assert all([text_history[-1].is_action for text_history in text_histories])

        prev_token_histories = []
        token_histories = []
        for text_history in text_histories:
            prev_token_histories.append(TokenHistory.from_text_history(text_history[:-1], tokenizer))
            token_histories.append(TokenHistory.from_text_history(text_history, tokenizer))
        
        # truncate to end and pad tokens
        tokens = np.stack([np.concatenate((token_history.tokens[-max_length:], np.full((max_length-min(token_history.tokens.shape[0], max_length),), tokenizer.pad_token_id)), axis=0) for token_history in token_histories], axis=0)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        # str_lst = [[]]
        
        all_logprobs = []
        #TODO: need attention mask
        # or just do from string thing
        for i in range(0, len(text_histories), bsize):
            tokens_batch = jnp.asarray(tokens[i:i+bsize, :])
    
            attention_mask = (tokens_batch != tokenizer.pad_token_id).astype(np.float32)
            
            # new_key = None
            # # if prng_key is not None:
            # prng_key, new_key = jax.random.split(prng_key)
            
            forward_batch_output = inference.forward(
                tokens_batch,
                attention_mask=attention_mask,
                train=False,
                prng_key=None,
            )
            # embed()
            policy_logits = forward_batch_output.policy_raw_output.logits
            prefix_len = jnp.asarray([prev_token_histories[i+x].tokens.shape[0] for x in range(tokens_batch.shape[0])], dtype=jnp.int32)
            action_logprobs = jnp.empty(prefix_len.shape, dtype=jnp.float32)
            
            logprobs = jax.nn.log_softmax(policy_logits, axis=-1)
            action_logits = jnp.take_along_axis(logprobs[:, :-1], tokens_batch[:, 1:][..., None], axis=2).squeeze(2)
            # trying to batchify
            masked_action_logits = action_logits * attention_mask[:, 1:]
            for x in range(len(prefix_len)):
                action_logprobs = action_logprobs.at[x].set(masked_action_logits[x][(prefix_len[x]-1):].sum(axis=0))
            # for x in range(len(prefix_len)):
            #     action_logprobs = action_logprobs.at[x].set((action_logits[x] * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))

            all_logprobs.extend(jax.device_get(action_logprobs).tolist())
        return all_logprobs

    return score_fn

def build_bc_score_fn(
    inference: Inference, 
    tokenizer: PreTrainedTokenizer, 
    max_length: int, 
    bsize: int, 
):
    
    def score_fn(text_histories: List[TextHistory]) -> List[float]:
        assert all([text_history[-1].is_action for text_history in text_histories])

        prev_token_histories = []
        token_histories = []
        for text_history in text_histories:
            prev_token_histories.append(TokenHistory.from_text_history(text_history[:-1], tokenizer))
            token_histories.append(TokenHistory.from_text_history(text_history, tokenizer))
        
        # truncate to end and pad tokens
        tokens = np.stack([np.concatenate((token_history.tokens[-max_length:], np.full((max_length-min(token_history.tokens.shape[0], max_length),), tokenizer.pad_token_id)), axis=0) for token_history in token_histories], axis=0)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        # str_lst = [[]]
        
        all_logprobs = []
        #TODO: need attention mask
        # or just do from string thing
        for i in range(0, len(text_histories), bsize):
            tokens_batch = jnp.asarray(tokens[i:i+bsize, :])
    
            attention_mask = (tokens_batch != tokenizer.pad_token_id).astype(np.float32)
            
            # new_key = None
            # # if prng_key is not None:
            # prng_key, new_key = jax.random.split(prng_key)
            
            forward_batch_output = inference.forward(
                tokens_batch,
                attention_mask=attention_mask,
                train=False,
                prng_key=None,
            )
            # embed()
            policy_logits = forward_batch_output.logits
            prefix_len = jnp.asarray([prev_token_histories[i+x].tokens.shape[0] for x in range(tokens_batch.shape[0])], dtype=jnp.int32)
            action_logprobs = jnp.empty(prefix_len.shape, dtype=jnp.float32)
            
            logprobs = jax.nn.log_softmax(policy_logits, axis=-1)
            action_logits = jnp.take_along_axis(logprobs[:, :-1], tokens_batch[:, 1:][..., None], axis=2).squeeze(2)
            # trying to batchify
            masked_action_logits = action_logits * attention_mask[:, 1:]
            for x in range(len(prefix_len)):
                action_logprobs = action_logprobs.at[x].set(masked_action_logits[x][(prefix_len[x]-1):].sum(axis=0))
            # for x in range(len(prefix_len)):
            #     action_logprobs = action_logprobs.at[x].set((action_logits[x] * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))

            all_logprobs.extend(jax.device_get(action_logprobs).tolist())
        return all_logprobs

    return score_fn

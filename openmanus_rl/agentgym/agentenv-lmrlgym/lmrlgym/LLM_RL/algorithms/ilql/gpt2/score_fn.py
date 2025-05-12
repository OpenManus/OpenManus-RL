from typing import List, Optional
from JaxSeq.models.gpt2.interface import GPT2Inference
from LLM_RL.algorithms.ilql.base_interface import ILQLInference
from transformers.tokenization_utils import PreTrainedTokenizer
import jax.numpy as jnp
import numpy as np
from LLM_RL.environment import TextHistory, TokenHistory
import jax
from IPython import embed

def build_ilql_score_fn(
    inference: ILQLInference, 
    pi_beta_inference: Optional[GPT2Inference], 
    tokenizer: PreTrainedTokenizer, 
    max_length: int, 
    value_weight: float, 
    logit_weight: Optional[float], 
    bsize: int, 
):
    assert (pi_beta_inference is None and logit_weight is None) or (pi_beta_inference is not None and logit_weight is not None)
    
    def score_fn(text_histories: List[TextHistory], done:Optional[List]=None) -> List[float]:
        assert all([text_history[-1].is_action for text_history in text_histories])
        
        prev_token_histories = []
        token_histories = []
        for text_history in text_histories:
            prev_token_histories.append(TokenHistory.from_text_history(text_history[:-1], tokenizer))
            token_histories.append(TokenHistory.from_text_history(text_history, tokenizer))
        
        # truncate to end and pad tokens
        tokens = np.stack([np.concatenate((token_history.tokens[-max_length:], np.full((max_length-min(token_history.tokens.shape[0], max_length),), tokenizer.pad_token_id)), axis=0) for token_history in token_histories], axis=0)
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        
        advantages = []
        
        for i in range(0, len(text_histories), bsize):
            batch = tokens[i:i+bsize, :]
            values = inference.forward(batch)
            # check prefix len is getting action
            prefix_len = jnp.asarray([prev_token_histories[i+x].tokens.shape[0] for x in range(batch.shape[0])], dtype=jnp.int32)
            attention_mask = (batch != tokenizer.pad_token_id).astype(np.float32)
            # embed()
            try:
                qs = jnp.minimum(values.target_output.q1, values.target_output.q2)
            except AttributeError:
                qs = jnp.minimum(values.q1, values.q2)
            qsa = jnp.take_along_axis(qs[:, :-1], batch[:, 1:][..., None], axis=2).squeeze(2)
            action_advs = jnp.empty(prefix_len.shape, dtype=jnp.float32)
            for x in range(len(prefix_len)):
                # embed()
                # check if this is getting rid of non-action states
                try:
                    action_advs = action_advs.at[x].set(value_weight * ((qsa[x] - values.output.v[x, :-1]) * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))
                except AttributeError:
                    action_advs = action_advs.at[x].set(value_weight * ((qsa[x] - values.v[x, :-1]) * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))

            if logit_weight is not None:
                logprobs = jax.nn.log_softmax(pi_beta_inference.get_logits_from_tokens(batch), axis=-1)
                action_logits = jnp.take_along_axis(logprobs[:, :-1], batch[:, 1:][..., None], axis=2).squeeze(2)
                for x in range(len(prefix_len)):
                    action_advs = action_advs.at[x].add(logit_weight * (action_logits[x] * attention_mask[x, 1:])[(prefix_len[x]-1):].sum(axis=0))

            advantages.extend(jax.device_get(action_advs).tolist())
        
        return advantages

    return score_fn

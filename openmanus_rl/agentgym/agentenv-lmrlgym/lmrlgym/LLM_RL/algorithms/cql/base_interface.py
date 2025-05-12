from __future__ import annotations
from typing import Union, Tuple, Any, Callable, Optional, NamedTuple, List
import jax
import jax.numpy as jnp
from jax.random import PRNGKeyArray
import optax
from LLM_RL.utils import get_tensor_stats
from flax import struct
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from jaxtyping import PyTree
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from transformers.modeling_flax_outputs import FlaxCausalLMOutput
from flax.core import freeze
from transformers.generation import GenerationConfig
import numpy as np
from JaxSeq.utils import block_sequences, BlockingStrategy, Padding, Truncation
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from JaxSeq.models.base_interface import GenerationFromStrOutput, Inference
from LLM_RL.environment import BatchedTextPolicy
from LLM_RL.algorithms.value_rl_base.base_interface import ValueRLForwardOutput, ValueRLInference
from LLM_RL.algorithms.ilql.base_interface import get_query_indicators
from LLM_RL.algorithms.ilql.base_interface import ILQLInference

# loss function

def cql_loss(
    q1: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    q2: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    target_q1: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    target_q2: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    target_q1_final: jax.Array, # [batch]
    target_q2_final: jax.Array, # [batch]
    q1_logits: jax.Array, # [batch, time-1, vocab] output is masked; shift x[:-1]
    q2_logits: jax.Array, # [batch, time-1, vocab] output is masked; shift x[:-1]
    token_ids: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    attention_mask: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    should_take_action: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    rewards: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    *, 
    gamma: Union[float, jax.Array], 
    cql_weight: Union[float, jax.Array], 
) -> Tuple[jnp.ndarray, Any]:
    # should be an action in the batch
    mask = should_take_action.astype(jnp.float32) * attention_mask
    n = mask.sum()
    
    q1sa_flat, q2sa_flat = q1.reshape(-1), q2.reshape(-1)
    target_q1nssa_flat = jnp.concatenate((target_q1, target_q1_final[..., None]), axis=1).reshape(-1)
    target_q2nssa_flat = jnp.concatenate((target_q2, target_q2_final[..., None]), axis=1).reshape(-1)

    q_query_indicators = get_query_indicators(should_take_action.reshape(-1))

    is_next_action = should_take_action.copy()
    # set first action position to false
    is_next_action = is_next_action.at[jnp.arange(0, is_next_action.shape[0], dtype=jnp.int32), jnp.argmax(is_next_action.astype(jnp.int32), axis=1)].set(False)
    # set endpoint to true as long as there is at least 1 action in the sequence
    is_next_action = jnp.concatenate((is_next_action, (should_take_action.sum(axis=1) > 0)[..., None]), axis=1)

    qns_query_indicators = get_query_indicators(is_next_action.reshape(-1))
    # should be the same number of qns as qv, so we can clip the extra padding to match shape
    qns_query_indicators = qns_query_indicators[:q_query_indicators.shape[0], :]
    
    # extract selected values
    q1sa_selected = (q_query_indicators * q1sa_flat).sum(axis=1)
    q2sa_selected = (q_query_indicators * q2sa_flat).sum(axis=1)
    target_q1nssa_selected = (qns_query_indicators * target_q1nssa_flat).sum(axis=1)
    target_q2nssa_selected = (qns_query_indicators * target_q2nssa_flat).sum(axis=1)
    rs_selected = (q_query_indicators * rewards.reshape(-1)).sum(axis=1)

    # get masks for selected values
    a_mask = (q_query_indicators.sum(axis=1) > 0).astype(jnp.float32)
    ans_mask = (qns_query_indicators.sum(axis=1) > 0).astype(jnp.float32)

    # target_qs
    target_qns_selected = jnp.minimum(target_q1nssa_selected, target_q2nssa_selected)

    # compute q loss
    q1_loss = (optax.l2_loss(q1sa_selected, jax.lax.stop_gradient(rs_selected + gamma * target_qns_selected)) * a_mask).sum() / n
    q2_loss = (optax.l2_loss(q2sa_selected, jax.lax.stop_gradient(rs_selected + gamma * target_qns_selected)) * a_mask).sum() / n

    # compute cql loss on both q heads
    q1_cql_loss = optax.softmax_cross_entropy_with_integer_labels(q1_logits, token_ids)
    q1_cql_loss = (mask * q1_cql_loss).sum() / n

    q2_cql_loss = optax.softmax_cross_entropy_with_integer_labels(q2_logits, token_ids)
    q2_cql_loss = (mask * q2_cql_loss).sum() / n
    
    loss = q1_loss + q2_loss + cql_weight * (q1_cql_loss + q2_cql_loss)

    logs = dict(
        losses=dict(
            total_loss=loss, 
            q1_loss=q1_loss, 
            q2_loss=q2_loss, 
            q1_cql_loss=q1_cql_loss, 
            q2_cql_loss=q2_cql_loss, 
        ), 
        q1=get_tensor_stats(q1sa_selected, mask=a_mask, n=n), 
        q2=get_tensor_stats(q2sa_selected, mask=a_mask, n=n), 
        target_qns=get_tensor_stats(target_qns_selected, mask=ans_mask, n=n), 
        target_q1ns=get_tensor_stats(target_q1nssa_selected, mask=ans_mask, n=n), 
        target_q2ns=get_tensor_stats(target_q2nssa_selected, mask=ans_mask, n=n), 
        rewards=get_tensor_stats(rewards, mask=mask, n=n), 
    )

    return loss, logs

class CQLTrain(struct.PyTreeNode):
    base_train_state: TrainState
    target_base_params: Optional[PyTree]
    q1_head_train_state: TrainState
    q2_head_train_state: TrainState
    q1_target_head_params: PyTree
    q2_target_head_params: PyTree
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     base_train_state: TrainState, 
    #     target_base_params: Optional[PyTree], 
    #     q1_head_train_state: TrainState, 
    #     q2_head_train_state: TrainState, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 

    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     rewards: jax.Array, 
    #     dones: jax.Array, 

    #     next_token_ids: Optional[jax.Array], 
    #     next_tokens_attention_mask: Optional[jax.Array], 
    #     next_tokens_position_ids: Optional[jax.Array], 
    #     next_dones: Optional[jax.Array], 

    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        rewards: jax.Array, # [batch, time-1]
        dones: jax.Array, # [batch]
        next_token_ids: Optional[jax.Array], # [batch, n_time]
        next_dones: Optional[jax.Array], # [batch]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        next_tokens_attention_mask: Optional[jax.Array]=None, 
        next_tokens_position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[CQLTrain, jax.Array, PyTree]:
        
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        if next_token_ids is not None:
            next_tokens_attention_mask, next_tokens_position_ids = initialize_attn_mask_pos_ids(
                next_token_ids, 
                self.tokenizer.pad_token_id, 
                next_tokens_attention_mask, 
                next_tokens_position_ids, 
            )
        else:
            assert next_tokens_attention_mask is None
            assert next_tokens_position_ids is None
        
        base_train_state, \
        target_base_params, \
        q1_head_train_state, \
        q2_head_train_state, \
        q1_target_head_params, \
        q2_target_head_params, \
        loss, logs = self._step(
            self.base_train_state, 
            self.target_base_params, 
            self.q1_head_train_state, 
            self.q2_head_train_state, 
            self.q1_target_head_params, 
            self.q2_target_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            rewards, 
            dones, 
            next_token_ids, 
            next_tokens_attention_mask, 
            next_tokens_position_ids, 
            next_dones, 
            prng_key, 
            train, 
        )

        return self.replace(
            base_train_state=base_train_state, 
            target_base_params=target_base_params, 
            q1_head_train_state=q1_head_train_state, 
            q2_head_train_state=q2_head_train_state, 
            q1_target_head_params=q1_target_head_params, 
            q2_target_head_params=q2_target_head_params, 
        ), loss, logs

class CQLInference(ILQLInference):
    # def _eval_loss(
    #     base_params: PyTree, 
    #     target_base_params: Optional[PyTree], 
    #     q1_head_params: PyTree, 
    #     q2_head_params: PyTree, 
    #     q1_target_head_params: PyTree, 
    #     q2_target_head_params: PyTree, 

    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     rewards: jax.Array, 
    #     dones: jax.Array, 

    #     next_token_ids: Optional[jax.Array], 
    #     next_tokens_attention_mask: Optional[jax.Array], 
    #     next_tokens_position_ids: Optional[jax.Array], 
    #     next_dones: Optional[jax.Array], 

    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def eval_loss(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        rewards: jax.Array, # [batch, time-1]
        dones: jax.Array, # [batch]
        next_token_ids: Optional[jax.Array], # [batch, n_time]
        next_dones: Optional[jax.Array], # [batch]
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        next_tokens_attention_mask: Optional[jax.Array]=None, 
        next_tokens_position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        if self.value_inference.q2_head_params is None:
            raise NotImplementedError
        if self.target_value_inference.q2_head_params is None:
            raise NotImplementedError

        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.value_inference.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        if next_token_ids is not None:
            next_tokens_attention_mask, next_tokens_position_ids = initialize_attn_mask_pos_ids(
                next_token_ids, 
                self.value_inference.tokenizer.pad_token_id, 
                next_tokens_attention_mask, 
                next_tokens_position_ids, 
            )
        else:
            assert next_tokens_attention_mask is None
            assert next_tokens_position_ids is None
        
        loss, logs = self._eval_loss(
            self.value_inference.base_params, 
            self.target_value_inference.base_params, 
            self.value_inference.q1_head_params, 
            self.value_inference.q2_head_params, 
            self.target_value_inference.q1_head_params, 
            self.target_value_inference.q2_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            rewards, 
            dones, 
            next_token_ids, 
            next_tokens_attention_mask, 
            next_tokens_position_ids, 
            next_dones, 
            prng_key, 
            train, 
        )

        return loss, logs
    
    def eval_loss_from_str(self, *args, **kwargs):
        raise NotImplementedError

from __future__ import annotations
from typing import Union, Tuple, Any, Callable, Optional
import jax
import jax.numpy as jnp
import optax
from LLM_RL.utils import get_tensor_stats
from flax import struct
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from jaxtyping import PyTree
from JaxSeq.models.base_interface import initialize_attn_mask_pos_ids
from LLM_RL.algorithms.ilql.base_interface import get_query_indicators, ValueRLInference
from IPython import embed

# loss function

def mc_loss(
    q: jax.Array, # [batch, time-1] output is masked; shift x[:-1]
    q_logits: jax.Array, # [batch, time-1, vocab] output is masked; shift x[:-1]
    token_ids: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    attention_mask: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    should_take_action: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    returns: jax.Array, # [batch, time-1] output is masked; shift x[1:]
    *, 
    cql_weight: Union[float, jax.Array], 
) -> Tuple[jnp.ndarray, Any]:
    # should be an action in the batch
    mask = should_take_action.astype(jnp.float32) * attention_mask
    n = mask.sum()
    q_query_indicators = get_query_indicators(should_take_action.reshape(-1))
    
    # extract selected values
    qsa_selected = (q_query_indicators * q.reshape(-1)).sum(axis=1)
    returns_selected = (q_query_indicators * returns.reshape(-1)).sum(axis=1)

    # get masks for selected values
    a_mask = (q_query_indicators.sum(axis=1) > 0).astype(jnp.float32)

    # compute q loss
    q_loss = (optax.l2_loss(qsa_selected, jax.lax.stop_gradient(returns_selected)) * a_mask).sum() / n

    # compute cql loss on both q heads
    q_cql_loss = optax.softmax_cross_entropy_with_integer_labels(q_logits, token_ids)
    q_cql_loss = (mask * q_cql_loss).sum() / n
    
    loss = q_loss + cql_weight * q_cql_loss

    logs = dict(
        losses=dict(
            total_loss=loss, 
            q_loss=q_loss, 
            q_cql_loss=q_cql_loss, 
        ), 
        q=get_tensor_stats(qsa_selected, mask=a_mask, n=n), 
        returns=get_tensor_stats(returns_selected, mask=a_mask, n=n), 
    )

    return loss, logs

class MCTrain(struct.PyTreeNode):
    base_train_state: TrainState
    q_head_train_state: TrainState
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _step: Callable = struct.field(pytree_node=False)
    
    # def _step(
    #     base_train_state: TrainState, 
    #     q_head_train_state: TrainState, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     returns: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray], 
    #     train: bool=True, 
    # ) -> Tuple[TrainState, TrainState, jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def step(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        returns: jax.Array, # [batch, time-1]
        prng_key: Optional[jax.random.PRNGKeyArray], 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        train: bool=True, 
    ) -> Tuple[MCTrain, jax.Array, PyTree]:
        
        # handle attention mask and position ids shifting
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )
        
        base_train_state, \
        q_head_train_state, \
        loss, logs = self._step(
            self.base_train_state, 
            self.q_head_train_state, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            returns, 
            prng_key, 
            train, 
        )

        return self.replace(
            base_train_state=base_train_state, 
            q_head_train_state=q_head_train_state, 
        ), loss, logs

class MCInference(ValueRLInference):
    _eval_loss: Callable = struct.field(pytree_node=False)

    # def _eval_loss(
    #     base_params: PyTree, 
    #     q_head_params: PyTree, 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     should_take_action: jax.Array, 
    #     returns: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     train: bool=False, 
    # ) -> Tuple[jax.Array, PyTree]:
    #     raise NotImplementedError
    
    def eval_loss(
        self, 
        input_ids: jax.Array, # [batch, time]
        should_take_action: jax.Array, # [batch, time-1]
        returns: jax.Array, # [batch, time-1]
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        train: bool=False, 
    ) -> Tuple[jax.Array, PyTree]:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )
        
        loss, logs = self._eval_loss(
            self.base_params, 
            self.q1_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            should_take_action, 
            returns, 
            prng_key, 
            train, 
        )

        return loss, logs
    
    def eval_loss_from_str(self, *args, **kwargs):
        raise NotImplementedError

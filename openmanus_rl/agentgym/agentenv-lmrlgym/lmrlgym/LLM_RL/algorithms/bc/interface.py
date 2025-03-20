from __future__ import annotations
from collections import namedtuple
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
from jax.experimental.maps import Mesh
import numpy as np
from jax.random import KeyArray
from optax import softmax_cross_entropy_with_integer_labels
from flax.core.frozen_dict import FrozenDict
import optax
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from environment import TextHistory, TokenHistory
from algorithms.jax_agent import Inference, StepOutput, Trainer
from JaxSeq.utils import block_sequences
from LLM_RL.algorithms.bc.data import block_token_histories
# from token_history import text_history_to_token_history
from transformers.tokenization_utils import PreTrainedTokenizer
from jax.experimental.pjit import pjit, with_sharding_constraint
from flax import struct
from jax.experimental import PartitionSpec

# loss function

def bc_loss(
    model: FlaxPreTrainedModel, 
    input_ids: jnp.ndarray, 
    attention_mask: jnp.ndarray, 
    is_action: jnp.ndarray, 
    params: PyTree, 
    rng: Optional[KeyArray], 
    train: bool, 
    *, 
    non_action_weight: Union[jnp.ndarray, float], 
) -> jnp.ndarray:
    logits = model(input_ids=input_ids, attention_mask=attention_mask, params=params, dropout_rng=rng, train=train).logits
    token_losses = softmax_cross_entropy_with_integer_labels(logits[:, :-1, :], input_ids[:, 1:]) * attention_mask[:, 1:]
    token_losses = is_action[:, 1:] * token_losses + (1 - is_action[:, 1:]) * token_losses * non_action_weight
    loss = token_losses.sum() / attention_mask[:, 1:].sum()
    return loss, {'loss': loss}

# main interface objects

class BCTrainer(Trainer):    
    def train_step_from_text_history(self, text_histories: List[TextHistory], max_len: Optional[int], rng_key: KeyArray) -> Tuple[jnp.ndarray, Dict[str, Any], BCTrainer]:

        token_histories = [TokenHistory.from_text_history(text_history, self.tokenizer) for text_history in text_histories]

        tokens, is_action = block_token_histories(token_histories, max_len, self.tokenizer.pad_token_id)
        
        loss, info, new_trainer = self.train_step(
            (jnp.asarray(tokens, dtype=jnp.int32), 
            jnp.asarray(is_action, dtype=jnp.bool_),), 
            rng_key, 
        )

        return loss, info, new_trainer

class BCInference(struct.PyTreeNode):
    logit_fn: Callable[[PyTree, jnp.ndarray], jnp.ndarray] = struct.field(pytree_node=False)
    loss_fn: Callable[[PyTree, PyTree], jnp.ndarray] = struct.field(pytree_node=False)
    
    def get_logits_from_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        
        logit_output = self.logit_fn(self.params, tokens)

        return logit_output
    
    def get_logits_from_str(self, strs: List[str], max_len: int) -> Tuple[jnp.ndarray, jnp.ndarray]:

        tokens = [self.tokenizer.encode(item) for item in strs]
        tokens = block_sequences(tokens, max_len=max_len, pad_value=self.tokenizer.pad_token_id, dtype=np.int32)

        logit_output = self.get_logits_from_tokens(
            jnp.asarray(tokens, dtype=jnp.int32), 
        )

        return logit_output, tokens
    
    def eval_loss(self, batch: PyTree) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        
        loss, info = self.loss_fn(self.params, batch)

        return loss, info
    
    def eval_loss_from_text_history(self, text_histories: List[TextHistory], max_len: int) -> Tuple[jnp.ndarray, Dict[str, Any]]:

        token_histories = [text_history_to_token_history(text_history, self.tokenizer) for text_history in text_histories]

        tokens, is_action = block_token_histories(token_histories, max_len, self.tokenizer.pad_token_id)

        loss, info = self.eval_loss(
            (jnp.asarray(tokens, dtype=jnp.int32), 
            jnp.asarray(is_action, dtype=jnp.bool_),), 
        )

        return loss, info

# load model parallel jax trainer

def load_bc_trainer(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    optim: optax.GradientTransformation, 
    optim_state: PyTree, 
    optim_state_spec: Optional[Any], 
    do_pjit: bool, 
    loss_fn: Callable[[FlaxPreTrainedModel, jnp.ndarray, jnp.ndarray, jnp.ndarray, PyTree, Optional[KeyArray], bool], Tuple[jnp.ndarray, Dict[str, Any]]], 
) -> BCTrainer:

    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)
    
    batch_spec = (PartitionSpec("dp", None), PartitionSpec("dp", None)) if do_pjit else None

    # define seq2seq training step
    def step_fn(params: PyTree, optim_state: PyTree, rng: KeyArray, batch: PyTree):
        # ensure it is sharded properly
        if do_pjit:
            batch = with_sharding_constraint(batch, batch_spec)
        tokens, is_action = batch
        attn_mask = (tokens != pad_id).astype(jnp.int32)
        is_action = is_action.astype(np.float32)
        def grad_loss(params: PyTree):
            loss, info = loss_fn(model, tokens, attn_mask, is_action, params, rng, True)
            return loss, info
        (loss, info), grads = jax.value_and_grad(grad_loss, has_aux=True)(params)
        if do_pjit:
            grads = with_sharding_constraint(grads, param_spec)
        updates, optim_state = optim.update(grads, optim_state, params)
        params = optax.apply_updates(params, updates)
        return StepOutput(loss, info, params, optim_state)

    if do_pjit:
        p_step_fn = pjit(
            step_fn, 
            in_axis_resources=(param_spec, optim_state_spec, None, batch_spec,), 
            out_axis_resources=StepOutput(None, None, param_spec, optim_state_spec), 
            donate_argnums=(0, 1), 
        )
    else:
        p_step_fn = step_fn
    
    train_interface = BCTrainer(params, optim_state, tokenizer, p_step_fn)

    return train_interface

# load model parallel jax inference

def load_bc_inference(
    model: FlaxPreTrainedModel, 
    params: PyTree, 
    param_spec: Optional[Any], 
    tokenizer: PreTrainedTokenizer, 
    do_pjit: bool, 
    loss_fn: Optional[Callable[[FlaxPreTrainedModel, jnp.ndarray, jnp.ndarray, jnp.ndarray, PyTree, Optional[KeyArray], bool], Tuple[jnp.ndarray, Dict[str, Any]]]], 
) -> BCInference:

    has_loss_fn = loss_fn is not None
    
    pad_id = jnp.asarray(tokenizer.pad_token_id, dtype=jnp.int32)
    
    batch_spec = (PartitionSpec("dp", None), PartitionSpec("dp", None)) if do_pjit else None
    tokens_spec = PartitionSpec("dp", None) if do_pjit else None
    logits_spec = PartitionSpec("dp", None, None) if do_pjit else None
    
    # define generation_fn
    def generate_fn(params: PyTree, rng: KeyArray, tokens: jnp.ndarray, kwargs: Dict[str, Any]) -> jnp.ndarray:
        if do_pjit:
            tokens = with_sharding_constraint(tokens, tokens_spec)
        attn_mask = (tokens != pad_id).astype(jnp.int32)
        out_sequences = model.generate(tokens, attention_mask=attn_mask, params=params, prng_key=rng, **kwargs).sequences
        if do_pjit:
            out_sequences = with_sharding_constraint(out_sequences, tokens_spec)
        return out_sequences
    
    if do_pjit:
        p_generate_fn = pjit(
            generate_fn, 
            in_axis_resources=(param_spec, None, tokens_spec), 
            out_axis_resources=tokens_spec, 
            static_argnums=(3,), 
        )
    else:
        p_generate_fn = generate_fn
    
    # define logit function
    def logit_fn(params: PyTree, tokens: jnp.ndarray) -> jnp.ndarray:
        if do_pjit:
            tokens = with_sharding_constraint(tokens, tokens_spec)
        attn_mask = (tokens != pad_id).astype(jnp.int32)
        logits = model(input_ids=tokens, attention_mask=attn_mask, params=params, train=False).logits
        if do_pjit:
            logits = with_sharding_constraint(logits, logits_spec)
        return logits
    
    if do_pjit:
        p_logit_fn = pjit(
            logit_fn, 
            in_axis_resources=(param_spec, tokens_spec,), 
            out_axis_resources=logits_spec, 
        )
    else:
        p_logit_fn = logit_fn
    
    # define eval loss
    def eval_loss_fn(params: PyTree, batch: PyTree) -> jnp.ndarray:
        if not has_loss_fn:
            raise NotImplementedError
        if do_pjit:
            batch = with_sharding_constraint(batch, batch_spec)
        tokens, is_action = batch
        attn_mask = (tokens != pad_id).astype(jnp.int32)
        is_action = is_action.astype(np.float32)
        loss, info = loss_fn(model, tokens, attn_mask, is_action, params, None, False)
        return loss, info
    
    if do_pjit and has_loss_fn:
        p_eval_loss_fn = pjit(
            eval_loss_fn, 
            in_axis_resources=(param_spec, batch_spec,), 
            out_axis_resources=None, 
        )
    else:
        p_eval_loss_fn = eval_loss_fn

    inference_inferface = BCInference(params, tokenizer, p_generate_fn, p_logit_fn, p_eval_loss_fn)

    return inference_inferface

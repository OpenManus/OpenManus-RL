from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils import ModelOutput
import jax.numpy as jnp
import flax.linen as nn
import jax
from flax.core.frozen_dict import freeze, unfreeze
from jax import lax
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from JaxSeq.stream_tokens import FlaxStreamGenerationMixin
from transformers.generation import FlaxGenerationMixin

@dataclass
class GPT2ValueRLGenerationOutput(ModelOutput):
    logits: jnp.ndarray = None
    past_key_values: Optional[Tuple[Tuple[Tuple[jnp.ndarray]]]] = None

class GPT2ValueRLGeneration(FlaxStreamGenerationMixin, FlaxGenerationMixin):
    
    def __init__(
        self, 
        base_model_config: PretrainedConfig, 
        pi_beta: Optional[FlaxPreTrainedModel], 
        value_base: FlaxPreTrainedModel, 
        q_head: nn.Module, 
        beta: Union[float, jnp.ndarray], 
    ):
        self.config = base_model_config
        self.pi_beta = pi_beta
        self.value_base = value_base
        self.q_head = q_head
        self.beta = beta
    
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None, 
        attention_mask: Optional[jnp.ndarray] = None, 
        params: dict = None, 
        past_key_values: Optional[Tuple[Tuple[Tuple[jnp.ndarray]]]] = None, 
        dropout_rng: jax.random.PRNGKey = None, 
        train: bool = False, 
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # pi_beta_params, q2_head_params is optional
        pi_beta_params, base_params, q1_head_params, q2_head_params = params
        assert ((pi_beta_params is None and self.pi_beta is None) or (pi_beta_params is not None and self.pi_beta is not None))
        has_pi_beta = pi_beta_params is not None

        pi_beta_past_kvs, base_past_kvs = None, None
        if past_key_values is not None:
            pi_beta_past_kvs, base_past_kvs = past_key_values
        
        if has_pi_beta:
            new_dropout_rng = None
            if dropout_rng is not None:
                dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            pi_beta_outputs = self.pi_beta(
                input_ids, 
                attention_mask=attention_mask, 
                past_key_values=pi_beta_past_kvs, 
                **kwargs, 
                params=pi_beta_params, 
                dropout_rng=new_dropout_rng, 
                train=train, 
            )
            pi_beta_logits = pi_beta_outputs.logits
            pi_beta_kvs = pi_beta_outputs.past_key_values
        else:
            pi_beta_logits = None
            pi_beta_kvs = None

        new_dropout_rng = None
        if dropout_rng is not None:
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        value_base_outputs = self.value_base(
            input_ids, 
            attention_mask=attention_mask, 
            past_key_values=base_past_kvs, 
            **kwargs, 
            output_hidden_states=True, 
            params=base_params, 
            dropout_rng=new_dropout_rng, 
            train=train, 
        )
        base_hidden_states = value_base_outputs.hidden_states[-1]
        base_kvs = value_base_outputs.past_key_values
        
        new_dropout_rng = None
        if dropout_rng is not None:
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        q1_logits = self.q_head.apply(
            freeze({'params': q1_head_params}), 
            base_hidden_states, 
            train=train, 
            rngs={'dropout': new_dropout_rng} if train else None, 
        )

        # q2 is optional
        if q2_head_params is not None:
            new_dropout_rng = None
            if dropout_rng is not None:
                dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            q2_logits = self.q_head.apply(
                freeze({'params': q2_head_params}), 
                base_hidden_states, 
                train=train, 
                rngs={'dropout': new_dropout_rng} if train else None, 
            )

            q_logits = jnp.minimum(q1_logits, q2_logits)
        else:
            q_logits = q1_logits

        if pi_beta_logits is not None:
            logits = pi_beta_logits + self.beta * q_logits
        else:
            logits = self.beta * q_logits

        return GPT2ValueRLGenerationOutput(logits=logits, past_key_values=(pi_beta_kvs, base_kvs,))
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length), dtype=jnp.int32)
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        if self.pi_beta is not None:
            init_variables_pi_beta = self.pi_beta.module.init(
                jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True, 
            )
            cache_pi_beta = unfreeze(init_variables_pi_beta["cache"])
        else:
            cache_pi_beta = None
        
        init_variables_base = self.value_base.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True, 
        )
        cache_base = unfreeze(init_variables_base["cache"])

        return cache_pi_beta, cache_base
    
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPT2 uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs
    
    def can_generate(self) -> bool:
        return True

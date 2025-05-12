from __future__ import annotations
from typing import Union, Callable, Optional, NamedTuple, List
import jax
import jax.numpy as jnp
from flax import struct
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
from JaxSeq.models.base_interface import GenerationFromStrOutput
from LLM_RL.environment import BatchedTextPolicy

class ValueRLForwardOutput(NamedTuple):
    base_raw_output: FlaxCausalLMOutput
    q1: jax.Array
    q2: Optional[jax.Array]
    v: Optional[jax.Array]

class ValueRLInference(struct.PyTreeNode):
    pi_beta_params: Optional[PyTree]
    base_params: PyTree
    q1_head_params: PyTree
    q2_head_params: Optional[PyTree]
    v_head_params: Optional[PyTree]
    pi_beta_model: Optional[FlaxPreTrainedModel] = struct.field(pytree_node=False)
    base_model: FlaxPreTrainedModel = struct.field(pytree_node=False)
    q_head_model: nn.Module = struct.field(pytree_node=False)
    v_head_model: Optional[nn.Module] = struct.field(pytree_node=False)
    tokenizer: PreTrainedTokenizerBase = struct.field(pytree_node=False)
    _generate: Callable = struct.field(pytree_node=False)
    _forward: Callable = struct.field(pytree_node=False)

    # def _generate(
    #     pi_beta_params: Optional[PyTree], 
    #     base_params: PyTree, 
    #     q1_head_params: PyTree, 
    #     q2_head_params: Optional[PyTree], 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     generation_config: Optional[FrozenDict]=None, 
    #     trace: bool=True, 
    # ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]
    
    # def _forward(
    #     base_params: PyTree, 
    #     q1_head_params: PyTree, 
    #     q2_head_params: Optional[PyTree], 
    #     v_head_params: Optional[PyTree], 
    #     input_ids: jax.Array, 
    #     attention_mask: jax.Array, 
    #     position_ids: jax.Array, 
    #     prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    #     output_attentions: Optional[bool]=None, 
    #     train: bool=False, 
    # ) -> ILQLSimpleForwardOutput:
    #     raise NotImplementedError

    def generate(
        self, 
        input_ids: jax.Array, 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        trace: bool=True, 
    ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._generate(
            self.pi_beta_params, 
            self.base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            freeze(generation_config.to_dict()) if generation_config is not None else None, 
            trace, 
        )
    
    def generate_from_str(
        self, 
        input_strs: List[str], 
        prng_key: Optional[jax.random.PRNGKeyArray], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        generation_config: Optional[GenerationConfig]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ) -> GenerationFromStrOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        if target_token_process is None:
            target_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # generate
        outputs = self.generate(
            jnp.asarray(tokens), 
            prng_key, 
            generation_config=generation_config, 
            trace=trace, 
        )
        # process outputs
        output_sequences = list(map(target_token_process, outputs.sequences.tolist()))
        output_scores = None
        if isinstance(outputs, FlaxBeamSearchOutput):
            output_scores = np.asarray(outputs.scores)
        # decode tokens
        output_strs = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return GenerationFromStrOutput(output_strs, output_scores)
    
    def forward(
        self, 
        input_ids: jax.Array, 
        attention_mask: Optional[jax.Array]=None, 
        position_ids: Optional[jax.Array]=None, 
        output_attentions: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    ) -> ValueRLForwardOutput:
        attention_mask, position_ids = initialize_attn_mask_pos_ids(
            input_ids, 
            self.tokenizer.pad_token_id, 
            attention_mask, 
            position_ids, 
        )

        return self._forward(
            self.base_params, 
            self.q1_head_params, 
            self.q2_head_params, 
            self.v_head_params, 
            input_ids, 
            attention_mask, 
            position_ids, 
            prng_key, 
            output_attentions, 
            train, 
        )
    
    def forward_from_str(
        self, 
        input_strs: List[str], 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.RIGHT, truncation=Truncation.RIGHT, max_length=None), 
        output_attentions: Optional[bool]=None, 
        output_hidden_states: Optional[bool]=None, 
        train: bool=False, 
        prng_key: Optional[jax.random.PRNGKeyArray]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> FlaxCausalLMOutput:
        if input_token_process is None:
            input_token_process = lambda x: x
        # tokenize
        tokens = [input_token_process(self.tokenizer.encode(item)) for item in input_strs]
        tokens = block_sequences(tokens, self.tokenizer.pad_token_id, np.int32, blocking_strategy)
        # forward
        outputs = self.forward(
            jnp.asarray(tokens), 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            train=train, 
            prng_key=prng_key, 
        )
        return outputs

class ValueRLPolicy(BatchedTextPolicy):
    def set_params(self, policy_params: PyTree) -> None:
        raise NotImplementedError

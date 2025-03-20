from typing import Optional, Callable, Union, List
from jax.experimental.pjit import pjit
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules, BlockingStrategy, Padding, Truncation
from functools import partial
import jax
from jax.sharding import PartitionSpec as PS
from jax.sharding import NamedSharding
from flax.core import FrozenDict
from transformers.generation import FlaxBeamSearchOutput, FlaxGreedySearchOutput, FlaxSampleOutput
from LLM_RL.algorithms.value_rl_base.gpt2.generation import GPT2ValueRLGeneration
from LLM_RL.algorithms.value_rl_base.base_interface import ValueRLForwardOutput, ValueRLInference
from JaxSeq.stream_tokens import StreamingGenerationConfig
from LLM_RL.algorithms.value_rl_base.base_interface import ValueRLPolicy
from transformers.generation import GenerationConfig
from LLM_RL.environment import TextHistory, Text, text_history_to_str
from transformers.modeling_flax_outputs import FlaxCausalLMOutputWithCrossAttentions
from JaxSeq.utils import strip_prompt_from_completion

class GPT2ValueRLInference(ValueRLInference):
    @classmethod
    def load_inference(
        cls, 
        pi_beta_params: Optional[PyTree], 
        base_params: PyTree, 
        q1_head_params: PyTree, 
        q2_head_params: Optional[PyTree], 
        v_head_params: Optional[PyTree], 
        pi_beta_model: Optional[FlaxPreTrainedModel], 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        v_head_model: Optional[nn.Module], 
        tokenizer: PreTrainedTokenizerBase, 
        beta: float=0.0, 
        dp_shard_logits: bool=True, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        if v_head_model is not None:
            assert mesh == v_head_model.config.mesh
        assert (pi_beta_model is None and pi_beta_params is None) or (pi_beta_model is not None and pi_beta_params is not None)
        
        pi_beta_params_partition_spec = PS() if pi_beta_params is None else match_partition_rules(pi_beta_model.config.get_partition_rules(), pi_beta_params)
        base_params_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_params)
        q1_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_head_params)
        q2_head_params_partition_spec = PS() if q2_head_params is None else match_partition_rules(q_head_model.config.get_partition_rules(), q2_head_params)
        v_head_params_partition_spec = PS() if v_head_params is None else match_partition_rules(v_head_model.config.get_partition_rules(), v_head_params)

        generator = None
        if pi_beta_model is not None:
            generator = GPT2ValueRLGeneration(
                base_model_config=base_model.config, 
                pi_beta=pi_beta_model, 
                value_base=base_model, 
                q_head=q_head_model, 
                beta=beta, 
            )

        if pi_beta_params is not None:
            @partial(
                pjit, 
                static_argnames=('generation_config', 'trace'), 
                in_shardings=(
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), pi_beta_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                    jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                    NamedSharding(mesh, PS()), 
                ), 
                out_shardings=NamedSharding(mesh, PS()), 
            )
            def _generate(
                pi_beta_params: Optional[PyTree], 
                base_params: PyTree, 
                q1_head_params: PyTree, 
                q2_head_params: Optional[PyTree], 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                # data parallel shard inputs
                input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
                attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
                position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))
                # NOTE: position_ids ignored by transformers

                # generate from model
                output = generator.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    params=(pi_beta_params, base_params, q1_head_params, q2_head_params), 
                    prng_key=prng_key, 
                    generation_config=StreamingGenerationConfig.from_dict(generation_config) if generation_config is not None else None, 
                    trace=trace, 
                )
                
                return output
        else:
            def _generate(
                pi_beta_params: Optional[PyTree], 
                base_params: PyTree, 
                q1_head_params: PyTree, 
                q2_head_params: Optional[PyTree], 
                input_ids: jax.Array, 
                attention_mask: jax.Array, 
                position_ids: jax.Array, 
                prng_key: Optional[jax.random.PRNGKeyArray]=None, 
                generation_config: Optional[FrozenDict]=None, 
                trace: bool=True, 
            ) -> Union[FlaxSampleOutput, FlaxGreedySearchOutput, FlaxBeamSearchOutput]:
                raise NotImplementedError
        
        @partial(
            pjit, 
            static_argnames=('output_attentions', 'train'), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=ValueRLForwardOutput(
                base_raw_output=FlaxCausalLMOutputWithCrossAttentions(
                    logits=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                    hidden_states=NamedSharding(mesh, PS()), # assume no sharding for hidden states
                    attentions=NamedSharding(mesh, PS()), # assume no sharding for attentions
                    cross_attentions=NamedSharding(mesh, PS()), # assume no sharding for cross attentions
                    past_key_values=NamedSharding(mesh, PS()), # assume no sharding for past key values
                ), 
                q1=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if dp_shard_logits else NamedSharding(mesh, PS()), 
                q2=NamedSharding(mesh, PS(("dp", "fsdp"), None, None)) if (dp_shard_logits and q2_head_params is not None) else NamedSharding(mesh, PS()), 
                v=NamedSharding(mesh, PS()), 
            ), 
        )
        def _forward(
            base_params: PyTree, 
            q1_head_params: PyTree, 
            q2_head_params: Optional[PyTree], 
            v_head_params: Optional[PyTree], 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            output_attentions: Optional[bool]=None, 
            train: bool=False, 
        ) -> ValueRLForwardOutput:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(("dp", "fsdp"), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(("dp", "fsdp"), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(("dp", "fsdp"), None))

            # get logits
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                train=train, 
                output_attentions=output_attentions, 
                output_hidden_states=True, 
                dropout_rng=new_key, 
            )
            # trunc padded logits
            base_output = base_output.replace(logits=base_output.logits.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf')))

            # get q1
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1 = q_head_model.apply(
                {'params': q1_head_params}, 
                base_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )
            # trunc padded qs
            q1 = q1.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))

            # get q2
            if q2_head_params is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q2 = q_head_model.apply(
                    {'params': q2_head_params}, 
                    base_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )
                # trunc padded qs
                q2 = q2.at[:, :, base_model.config.unpadded_vocab_size:].set(-float('inf'))
            else:
                q2 = None

            if v_head_params is not None:
                # get v
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                v = v_head_model.apply(
                    {'params': v_head_params}, 
                    base_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                ).squeeze(2)
            else:
                v = None

            # assert sharding on outputs
            if dp_shard_logits:
                base_output = base_output.replace(logits=with_named_sharding_constraint(base_output.logits, mesh, PS(("dp", "fsdp"), None, None)))
                q1 = with_named_sharding_constraint(q1, mesh, PS(("dp", "fsdp"), None, None))
                if q2 is not None:
                    q2 = with_named_sharding_constraint(q2, mesh, PS(("dp", "fsdp"), None, None))
            return ValueRLForwardOutput(
                base_raw_output=base_output, 
                q1=q1, 
                q2=q2, 
                v=v, 
            )

        return cls(
            pi_beta_params=pi_beta_params, 
            base_params=base_params, 
            q1_head_params=q1_head_params, 
            q2_head_params=q2_head_params, 
            v_head_params=v_head_params, 
            pi_beta_model=pi_beta_model, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            v_head_model=v_head_model, 
            tokenizer=tokenizer, 
            _generate=_generate, 
            _forward=_forward,
        )

class GPT2ValuePolicy(ValueRLPolicy):
    def __init__(
        self, 
        inference: ValueRLInference, 
        prng_key: Optional[jax.random.KeyArray], 
        generation_config: Optional[GenerationConfig]=None, 
        blocking_strategy: BlockingStrategy=BlockingStrategy(padding=Padding.LEFT, truncation=Truncation.LEFT, max_length=None), 
        in_str_process: Optional[Callable[[str], str]]=None, 
        out_str_process: Optional[Callable[[str], str]]=None, 
        input_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        target_token_process: Optional[Callable[[List[int]], List[int]]]=None, 
        trace: bool=True, 
    ):
        self.inference = inference
        self.prng_key = prng_key
        self.generation_config = generation_config
        self.blocking_strategy = blocking_strategy
        self.in_str_process = in_str_process
        self.out_str_process = out_str_process
        self.input_token_process = input_token_process
        self.target_token_process = target_token_process
        if self.in_str_process is None:
            self.in_str_process = lambda x: x
        if self.out_str_process is None:
            self.out_str_process = lambda x: x
        self.trace = trace
    
    def act(self, text_history: List[Optional[TextHistory]], done: Optional[List[bool]]=None) -> List[Optional[TextHistory]]:
        if done is None:
            done = [False]*len(text_history)
        # force eos_token for done sequences
        eos_token = self.inference.tokenizer.eos_token
        if self.generation_config is not None and self.generation_config.eos_token_id is not None:
            eos_token = self.inference.tokenizer.decode(self.generation_config.eos_token_id)
        if eos_token is None:
            eos_token = self.inference.tokenizer.pad_token
        if eos_token is None:
            eos_token = ''
        
        raw_input_strs = [
            eos_token if d else self.in_str_process(text_history_to_str(item)) \
                for item, d in zip(text_history, done)
        ]

        new_key = None
        if self.prng_key is not None:
            self.prng_key, new_key = jax.random.split(self.prng_key)
        model_outputs = self.inference.generate_from_str(
            input_strs=raw_input_strs, 
            prng_key=new_key, 
            blocking_strategy=self.blocking_strategy, 
            generation_config=self.generation_config, 
            input_token_process=self.input_token_process, 
            target_token_process=self.target_token_process, 
            trace=self.trace, 
        )

        raw_output_strs = model_outputs.output_strs
        output_strs = [
            "" if d else self.out_str_process(strip_prompt_from_completion(raw_input_str, raw_output_str)) \
                for raw_input_str, raw_output_str, d in zip(raw_input_strs, raw_output_strs, done)
        ]

        return [
            None if d else text_history_item+(Text(output_str, True),) \
                for text_history_item, output_str, d in zip(text_history, output_strs, done)
        ]
    
    def set_params(self, policy_params: PyTree) -> None:
        pi_beta_params, base_params, \
            q1_head_params, q2_head_params = policy_params
        self.inference = self.inference.replace(
            pi_beta_params=pi_beta_params, 
            base_params=base_params, 
            q1_head_params=q1_head_params, 
            q2_head_params=q2_head_params, 
        )

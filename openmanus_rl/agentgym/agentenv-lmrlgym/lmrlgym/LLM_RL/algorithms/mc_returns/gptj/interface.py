from typing import Optional, Callable, Tuple
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.mc_returns.base_interface import MCTrain, MCInference
from flax.training.train_state import TrainState
from jaxtyping import PyTree
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizerBase
import flax.linen as nn
from JaxSeq.utils import with_named_sharding_constraint, match_partition_rules
from functools import partial
import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
import jax.numpy as jnp
from LLM_RL.algorithms.ilql.gptj.interface import GPTJValueRLInference


class GPTJMCTrain(MCTrain):
    @classmethod
    def load_train(
        cls, 
        base_train_state: TrainState, 
        q_head_train_state: TrainState, 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        detach_q: bool, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        base_train_state_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_train_state)
        q_head_train_state_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q_head_train_state)

        @partial(
            pjit, 
            donate_argnums=(0, 1), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q_head_train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q_head_train_state_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            base_train_state: TrainState, 
            q_head_train_state: TrainState, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            returns: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            returns = with_named_sharding_constraint(returns, mesh, PS(('dp', 'fsdp'), None))

            # define loss function

            def grad_loss(base_params: PyTree, q_head_params: PyTree, prng_key: jax.random.PRNGKeyArray):
                
                # get base hidden states

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                base_model_output = base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
                
                # get values

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q_head_output = q_head_model.apply(
                    {'params': q_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                # stop gradients
                if detach_q:
                    q_head_output = jax.lax.stop_gradient(q_head_output)

                q = jnp.take_along_axis(q_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                q_logits = q_head_output[:, :-1, :].astype(jnp.float32)

                loss, info = loss_fn(
                    q, 
                    q_logits, 
                    input_ids[:, 1:], 
                    attention_mask[:, 1:], 
                    should_take_action, 
                    returns, 
                )
                return loss, info

            # take loss
            (loss, info), (base_grads, q_head_grads) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0, 1))(
                base_train_state.params, 
                q_head_train_state.params, 
                prng_key, 
            )
            # assert shard gradients
            base_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                base_grads, 
                base_train_state_partition_spec.params, 
            )
            q_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                q_head_grads, 
                q_head_train_state_partition_spec.params, 
            )
            # update params and optim state
            base_train_state = base_train_state.apply_gradients(grads=base_grads)
            q_head_train_state = q_head_train_state.apply_gradients(grads=q_head_grads)

            return base_train_state, q_head_train_state, loss, info

        return cls(
            base_train_state=base_train_state, 
            q_head_train_state=q_head_train_state, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )

class GPTJMCInference(MCInference):
    @classmethod
    def load_inference(
        cls, 
        pi_beta_params: Optional[PyTree], 
        base_params: PyTree, 
        q_head_params: PyTree, 
        pi_beta_model: Optional[FlaxPreTrainedModel], 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        beta: float=0.0, 
        dp_shard_logits: bool=True, 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh

        value_inference = GPTJValueRLInference.load_inference(
            pi_beta_params=pi_beta_params, 
            base_params=base_params, 
            q1_head_params=q_head_params, 
            q2_head_params=None, 
            v_head_params=None, 
            pi_beta_model=pi_beta_model, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            v_head_model=None, 
            tokenizer=tokenizer, 
            beta=beta, 
            dp_shard_logits=dp_shard_logits, 
        )

        base_params_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_params)
        q_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q_head_params)

        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _eval_loss(
            base_params: TrainState, 
            q_head_params: TrainState, 
            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            returns: jax.Array, 
            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ):
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            returns = with_named_sharding_constraint(returns, mesh, PS(('dp', 'fsdp'), None))

            # get base hidden states
            
            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_model_output = base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
            )
            
            # get values

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q_head_output = q_head_model.apply(
                {'params': q_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            q = jnp.take_along_axis(q_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            q_logits = q_head_output[:, :-1, :].astype(jnp.float32)

            loss, info = loss_fn(
                q, 
                q_logits, 
                input_ids[:, 1:], 
                attention_mask[:, 1:], 
                should_take_action, 
                returns, 
            )

            return loss, info
    
        return cls(
            pi_beta_params=value_inference.pi_beta_params, 
            base_params=value_inference.base_params, 
            q1_head_params=value_inference.q1_head_params, 
            q2_head_params=value_inference.q2_head_params, 
            v_head_params=value_inference.v_head_params, 
            pi_beta_model=value_inference.pi_beta_model, 
            base_model=value_inference.base_model, 
            q_head_model=value_inference.q_head_model, 
            v_head_model=value_inference.v_head_model, 
            tokenizer=value_inference.tokenizer, 
            _generate=value_inference._generate, 
            _forward=value_inference._forward, 
            _eval_loss=_eval_loss, 
        )

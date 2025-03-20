from typing import Optional, Callable, Tuple, List
from jax.experimental.pjit import pjit
from LLM_RL.algorithms.cql.base_interface import CQLTrain, CQLInference
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
import optax
from LLM_RL.algorithms.value_rl_base.gptj.interface import GPTJValueRLInference

class GPTJCQLTrain(CQLTrain):
    @classmethod
    def load_train(
        cls, 
        base_train_state: TrainState, 
        target_base_params: Optional[PyTree], 
        q1_head_train_state: TrainState, 
        q2_head_train_state: TrainState, 
        q1_target_head_params: PyTree, 
        q2_target_head_params: PyTree, 
        base_model: FlaxPreTrainedModel, 
        q_head_model: nn.Module, 
        tokenizer: PreTrainedTokenizerBase, 
        loss_fn: Callable, 
        detach_q1: bool, 
        detach_q2: bool, 
        detach_v: bool, 
        polyak_alpha: float, 
        hard_update_every: Optional[int], 
    ):
        mesh = base_model.config.mesh
        assert mesh is not None
        assert mesh == q_head_model.config.mesh
        base_train_state_partition_spec = match_partition_rules(base_model.config.get_partition_rules(), base_train_state)
        target_base_params_partition_spec = PS() if target_base_params is None else match_partition_rules(base_model.config.get_partition_rules(), target_base_params)
        q1_head_train_state_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_head_train_state)
        q2_head_train_state_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_head_train_state)
        q1_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q1_target_head_params)
        q2_target_head_params_partition_spec = match_partition_rules(q_head_model.config.get_partition_rules(), q2_target_head_params)

        @partial(
            pjit, 
            donate_argnums=(0, 1, 2, 3, 4, 5), 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
            out_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_train_state_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
            ), 
        )
        def _step(
            base_train_state: TrainState, 
            target_base_params: Optional[PyTree], 
            q1_head_train_state: TrainState, 
            q2_head_train_state: TrainState, 
            q1_target_head_params: PyTree, 
            q2_target_head_params: PyTree, 

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray], 
            train: bool=True, 
        ) -> Tuple[TrainState, Optional[PyTree], TrainState, TrainState, TrainState, PyTree, PyTree, jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            dones = with_named_sharding_constraint(dones, mesh, PS(('dp', 'fsdp')))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS(('dp', 'fsdp'), None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS(('dp', 'fsdp')))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None

            # define loss function

            def grad_loss(base_params: PyTree, q1_head_params: PyTree, q2_head_params: PyTree, prng_key: jax.random.PRNGKeyArray):
                
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

                if target_base_params is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    target_base_model_output = base_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        position_ids=position_ids, 
                        params=target_base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=True, 
                    )
                else:
                    target_base_model_output = base_model_output
                
                if next_token_ids is not None:
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    next_token_base_model_output = base_model(
                        input_ids=next_token_ids, 
                        attention_mask=next_tokens_attention_mask, 
                        position_ids=next_tokens_position_ids, 
                        params=base_params, 
                        dropout_rng=new_key, 
                        train=train, 
                        output_hidden_states=True, 
                    )
                
                # get values

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q1_head_output = q_head_model.apply(
                    {'params': q1_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                q2_head_output = q_head_model.apply(
                    {'params': q2_head_params}, 
                    base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_q1_head_output = q_head_model.apply(
                    {'params': q1_target_head_params}, 
                    target_base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_q2_head_output = q_head_model.apply(
                    {'params': q2_target_head_params}, 
                    target_base_model_output.hidden_states[-1], 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                )

                # stop gradients
                if detach_q1:
                    q1_head_output = jax.lax.stop_gradient(q1_head_output)
                if detach_q2:
                    q2_head_output = jax.lax.stop_gradient(q2_head_output)
                if detach_v:
                    v_head_output = jax.lax.stop_gradient(v_head_output)
                target_q1_head_output = jax.lax.stop_gradient(target_q1_head_output)
                target_q2_head_output = jax.lax.stop_gradient(target_q2_head_output)

                q1 = jnp.take_along_axis(q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                q2 = jnp.take_along_axis(q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                # v = v_head_output[:, :-1].squeeze(2)
                # v_full = v_head_output.squeeze(2)
                target_q1 = jnp.take_along_axis(target_q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
                target_q2 = jnp.take_along_axis(target_q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)

                q1_logits = q1_head_output[:, :-1, :].astype(jnp.float32)
                q2_logits = q2_head_output[:, :-1, :].astype(jnp.float32)

                # get next token values

                if next_token_ids is not None:
                    # just run vf on last token to save some flops
                    last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_next_token_h = next_token_base_model_output.hidden_states[-1][jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]
                    new_key = None
                    if prng_key is not None:
                        prng_key, new_key = jax.random.split(prng_key)
                    next_token_v_head_output = q_head_model.apply(
                        {'params': q1_target_head_params}, 
                        final_next_token_h, 
                        train=train, 
                        rngs={'dropout': new_key} if prng_key is not None else None, 
                    )
                    v_final = next_token_v_head_output * (1 - next_dones.astype(jnp.float32))
                else:
                    last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                    last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                    final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                    v_final = v_full[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                    v_final = v_final * (1 - dones)
                v_final = jax.lax.stop_gradient(v_final)

                loss, info = loss_fn(
                    q1, 
                    q2, 
                    v, 
                    v_final, 
                    target_q1, 
                    target_q2, 
                    q1_logits, 
                    q2_logits, 
                    input_ids[:, 1:], 
                    attention_mask[:, 1:], 
                    should_take_action, 
                    rewards, 
                )
                return loss, info

            # take loss
            (loss, info), (base_grads, q1_head_grads, q2_head_grads) = jax.value_and_grad(grad_loss, has_aux=True, argnums=(0, 1, 2))(
                base_train_state.params, 
                q1_head_train_state.params, 
                q2_head_train_state.params, 
                prng_key, 
            )
            # assert shard gradients
            base_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                base_grads, 
                base_train_state_partition_spec.params, 
            )
            q1_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                q1_head_grads, 
                q1_head_train_state_partition_spec.params, 
            )
            q2_head_grads = jax.tree_util.tree_map(
                lambda x, ps: with_named_sharding_constraint(x, mesh, ps), 
                q2_head_grads, 
                q2_head_train_state_partition_spec.params, 
            )
            # update params and optim state
            base_train_state = base_train_state.apply_gradients(grads=base_grads)
            q1_head_train_state = q1_head_train_state.apply_gradients(grads=q1_head_grads)
            q2_head_train_state = q2_head_train_state.apply_gradients(grads=q2_head_grads)

            # handle target network updates
            def update_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                base_params = optax.incremental_update(params, base_params, polyak_alpha)
                if hard_update_every is not None:
                    base_params = optax.periodic_update(params, base_params, steps, hard_update_every)
                return base_params
            
            def mid_targets(params: PyTree, base_params: PyTree, steps: jnp.ndarray) -> PyTree:
                return base_params

            def update_cond(opt_state: PyTree) -> bool:
                if hasattr(opt_state, 'mini_step'):
                    return opt_state.mini_step == 0
                return True
            
            if target_base_params is not None:
                target_base_params = jax.lax.cond(
                    update_cond(base_train_state.opt_state), 
                    update_targets, 
                    mid_targets, 
                    base_train_state.params, 
                    target_base_params, 
                    base_train_state.step, 
                )
            q1_target_head_params = jax.lax.cond(
                update_cond(q1_head_train_state.opt_state), 
                update_targets, 
                mid_targets, 
                q1_head_train_state.params, 
                q1_target_head_params, 
                q1_head_train_state.step, 
            )
            q2_target_head_params = jax.lax.cond(
                update_cond(q2_head_train_state.opt_state), 
                update_targets, 
                mid_targets, 
                q2_head_train_state.params, 
                q2_target_head_params, 
                q2_head_train_state.step, 
            )

            return base_train_state, target_base_params, q1_head_train_state, q2_head_train_state, q1_target_head_params, q2_target_head_params, loss, info

        return cls(
            base_train_state=base_train_state, 
            target_base_params=target_base_params, 
            q1_head_train_state=q1_head_train_state, 
            q2_head_train_state=q2_head_train_state, 
            q1_target_head_params=q1_target_head_params, 
            q2_target_head_params=q2_target_head_params, 
            base_model=base_model, 
            q_head_model=q_head_model, 
            tokenizer=tokenizer, 
            _step=_step, 
        )

class GPTJILQLInference(ILQLInference):
    @classmethod
    def load_inference(
        cls, 
        value_inference: GPTJValueRLInference, 
        target_value_inference: GPTJValueRLInference, 
        loss_fn: Callable, 
    ):
        mesh = value_inference.base_model.config.mesh
        assert mesh is not None
        assert mesh == value_inference.q_head_model.config.mesh
        assert mesh == value_inference.v_head_model.config.mesh
        assert mesh == target_value_inference.base_model.config.mesh
        assert mesh == target_value_inference.q_head_model.config.mesh

        base_params_partition_spec = match_partition_rules(value_inference.base_model.config.get_partition_rules(), value_inference.base_params)
        target_base_params_partition_spec = match_partition_rules(target_value_inference.base_model.config.get_partition_rules(), target_value_inference.base_params)
        q1_head_params_partition_spec = match_partition_rules(value_inference.q_head_model.config.get_partition_rules(), value_inference.q1_head_params)
        q2_head_params_partition_spec = match_partition_rules(value_inference.q_head_model.config.get_partition_rules(), value_inference.q2_head_params)
        v_head_params_partition_spec = match_partition_rules(value_inference.v_head_model.config.get_partition_rules(), value_inference.v_head_params)
        q1_target_head_params_partition_spec = match_partition_rules(target_value_inference.q_head_model.config.get_partition_rules(), target_value_inference.q1_head_params)
        q2_target_head_params_partition_spec = match_partition_rules(target_value_inference.q_head_model.config.get_partition_rules(), target_value_inference.q2_head_params)
        
        @partial(
            pjit, 
            static_argnames=('train',), 
            in_shardings=(
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), target_base_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), v_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q1_target_head_params_partition_spec), 
                jax.tree_util.tree_map(lambda ps: NamedSharding(mesh, ps), q2_target_head_params_partition_spec), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
                NamedSharding(mesh, PS()), 
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
            base_params: PyTree, 
            target_base_params: Optional[PyTree], 
            q1_head_params: PyTree, 
            q2_head_params: PyTree, 
            v_head_params: PyTree, 
            q1_target_head_params: PyTree, 
            q2_target_head_params: PyTree, 

            input_ids: jax.Array, 
            attention_mask: jax.Array, 
            position_ids: jax.Array, 
            should_take_action: jax.Array, 
            rewards: jax.Array, 
            dones: jax.Array, 

            next_token_ids: Optional[jax.Array], 
            next_tokens_attention_mask: Optional[jax.Array], 
            next_tokens_position_ids: Optional[jax.Array], 
            next_dones: Optional[jax.Array], 

            prng_key: Optional[jax.random.PRNGKeyArray]=None, 
            train: bool=False, 
        ) -> Tuple[jax.Array, PyTree]:
            # data parallel shard inputs
            input_ids = with_named_sharding_constraint(input_ids, mesh, PS(('dp', 'fsdp'), None))
            attention_mask = with_named_sharding_constraint(attention_mask, mesh, PS(('dp', 'fsdp'), None))
            position_ids = with_named_sharding_constraint(position_ids, mesh, PS(('dp', 'fsdp'), None))
            should_take_action = with_named_sharding_constraint(should_take_action, mesh, PS(('dp', 'fsdp'), None))
            rewards = with_named_sharding_constraint(rewards, mesh, PS(('dp', 'fsdp'), None))
            dones = with_named_sharding_constraint(dones, mesh, PS(('dp', 'fsdp')))
            if next_token_ids is not None:
                assert next_tokens_attention_mask is not None
                assert next_tokens_position_ids is not None
                next_token_ids = with_named_sharding_constraint(next_token_ids, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_attention_mask = with_named_sharding_constraint(next_tokens_attention_mask, mesh, PS(('dp', 'fsdp'), None))
                next_tokens_position_ids = with_named_sharding_constraint(next_tokens_position_ids, mesh, PS(('dp', 'fsdp'), None))
                next_dones = with_named_sharding_constraint(next_dones, mesh, PS(('dp', 'fsdp')))
            else:
                assert next_tokens_attention_mask is None
                assert next_tokens_position_ids is None
                
            # get base hidden states

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            base_model_output = value_inference.base_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                params=base_params, 
                dropout_rng=new_key, 
                train=train, 
                output_hidden_states=True, 
            )

            if target_base_params is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                target_base_model_output = target_value_inference.base_model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    params=target_base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            else:
                target_base_model_output = base_model_output
            
            if next_token_ids is not None:
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                next_token_base_model_output = value_inference.base_model(
                    input_ids=next_token_ids, 
                    attention_mask=next_tokens_attention_mask, 
                    position_ids=next_tokens_position_ids, 
                    params=base_params, 
                    dropout_rng=new_key, 
                    train=train, 
                    output_hidden_states=True, 
                )
            
            # get values

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q1_head_output = value_inference.q_head_model.apply(
                {'params': q1_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            q2_head_output = value_inference.q_head_model.apply(
                {'params': q2_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            v_head_output = value_inference.v_head_model.apply(
                {'params': v_head_params}, 
                base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            target_q1_head_output = target_value_inference.q_head_model.apply(
                {'params': q1_target_head_params}, 
                target_base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            new_key = None
            if prng_key is not None:
                prng_key, new_key = jax.random.split(prng_key)
            target_q2_head_output = target_value_inference.q_head_model.apply(
                {'params': q2_target_head_params}, 
                target_base_model_output.hidden_states[-1], 
                train=train, 
                rngs={'dropout': new_key} if prng_key is not None else None, 
            )

            # process outputs

            q1 = jnp.take_along_axis(q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            q2 = jnp.take_along_axis(q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            v = v_head_output[:, :-1].squeeze(2)
            v_full = v_head_output.squeeze(2)
            target_q1 = jnp.take_along_axis(target_q1_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)
            target_q2 = jnp.take_along_axis(target_q2_head_output[:, :-1], input_ids[:, 1:][..., None], axis=2).squeeze(2)

            q1_logits = q1_head_output[:, :-1, :].astype(jnp.float32)
            q2_logits = q2_head_output[:, :-1, :].astype(jnp.float32)

            # get next token values

            if next_token_ids is not None:
                # just run vf on last token to save some flops
                last_next_token_idxs = (next_tokens_attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(next_tokens_attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_next_token_h = next_token_base_model_output.hidden_states[-1][jnp.arange(0, input_ids.shape[0], dtype=jnp.int32), last_next_token_idxs, :]
                new_key = None
                if prng_key is not None:
                    prng_key, new_key = jax.random.split(prng_key)
                next_token_v_head_output = value_inference.v_head_model.apply(
                    {'params': v_head_params}, 
                    final_next_token_h, 
                    train=train, 
                    rngs={'dropout': new_key} if prng_key is not None else None, 
                ).squeeze(1)
                v_final = next_token_v_head_output * (1 - next_dones.astype(jnp.float32))
            else:
                last_action_idxs = (should_take_action.shape[1]-1)-jnp.argmax(jnp.flip(should_take_action, axis=1).astype(jnp.int32), axis=1)+1
                last_token_idxs = (attention_mask.shape[1]-1)-jnp.argmax(jnp.flip(attention_mask, axis=1).astype(jnp.int32), axis=1)
                final_state_idxs = ((1 - dones) * last_action_idxs + dones * last_token_idxs).astype(jnp.int32)
                v_final = v_full[jnp.arange(0, should_take_action.shape[0], dtype=jnp.int32), final_state_idxs]
                v_final = v_final * (1 - dones)

            loss, info = loss_fn(
                q1, 
                q2, 
                v, 
                v_final, 
                target_q1, 
                target_q2, 
                q1_logits, 
                q2_logits, 
                input_ids[:, 1:], 
                attention_mask[:, 1:], 
                should_take_action, 
                rewards, 
            )
            
            return loss, info

        return cls(
            value_inference=value_inference, 
            target_value_inference=target_value_inference, 
            _eval_loss=_eval_loss, 
        )

import contextlib
from typing import Optional
import jax
from algorithms.jax_agent import Inference, JaxAgentPolicy, ReRankerPolicy, build_agent_proposer
from algorithms.jax_ilql.core import ilql_loss, load_ilql_inference, load_ilql_trainer
from algorithms.jax_ilql.models import BaseModelState, load_advanced_mlp, load_advanced_mlp2, load_advanced_mlp3, load_advanced_mlp4, load_linear, load_mlp
from environment import Text, TextTransition
from jax_models.gptj import load_gptj_model
from jax_utils.jax_shard import OptimType, shard_params
from jax_models.gpt2 import load_gpt2_model
import numpy as np
from jax.experimental.maps import Mesh
import optax
import tyro
from functools import partial
from text_env_eval import text_env_eval
from token_history import PrefixTokenTrajectory, text_transition_to_token_transition
from utils.path import convert_path
import os
import pickle as pkl
from algorithms.jax_ilql.data import ILQLDataset, ILQLIterableDataset
from algorithms.jax_ilql.load_ilql import load_ilql_default_train
from transformers import AutoTokenizer
from environments.car_dealer.data import create_trajectories_from_conversations, Role
import tree
import json
from algorithms.jax_ilql.basic_train_loop import eval_loop, train_loop
from utils.gcs_manager import open_pp as open


def main(
    exp_name: Optional[str], 
    model_name: str, 

    /,  # Mark the end of positional arguments.
    
    role: Role=Role.SELLER, 
    reward_mode: str="fancy", 

    lm_checkpoint_path: Optional[str]=None, 
    value_checkpoint_path: Optional[str]=None, 
    value_checkpoint_idx: Optional[int]=None, 
    checkpoint_is_sharded: bool=True, 
    init_value_with_lm: bool=False,

    data_path: Optional[str]='data/car_dealer', 
    output_path: Optional[str]='outputs/car_dealer', 

    use_wandb: bool=False, 
    wandb_project: Optional[str]='car_dealer-ilql', 

    do_pjit: bool=True, 
    model_p_shape: int=1, 
    data_p_shape: int=1, 

    epochs: int=2, 
    max_steps: Optional[int]=None, 
    eval_batches: Optional[int]=None, 
    
    use_adafactor: bool=False, 
    lr: float=5e-5, 
    weight_decay: float=0.0, 

    tau: float=0.5,
    cql_weight: float=0.1,

    train_bsize: int=32, 
    grad_accum_steps: int=1, 

    gradient_checkpoint: bool=True, 

    max_sequence_length: int=1024, 

    log_every: Optional[int]=None, 
    num_logs_per_epoch: float=10,
    eval_every: Optional[int]=None,
    num_evals_per_epoch: float=2,
    save_every: Optional[int]=None, 
    num_saves_per_epoch: float=1,
    save_best: bool=False,
    save_best_also: bool=False,
    save_last: bool=False,

    inference_bsize: int=32,
    inference_beta: Optional[float]=None,

    seed: int=0,

    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[str]=None, 

):
    if use_adafactor:
        assert weight_decay == 0.0, 'no weight decay with adafactor'
    if gcloud_project is not None and gcloud_token is None:
        gcloud_token = os.path.join(os.path.expanduser('~'), f'.config/gcloud/{gcloud_project}.json')

    input_args = locals().copy()
    print(input_args)

    
    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("set pad_token")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    with open(convert_path(os.path.join(data_path, 'train.json')), 'r') as f:
        raw_train = json.load(f)
    with open(convert_path(os.path.join(data_path, 'eval.json')), 'r') as f:
        raw_eval = json.load(f)

    train_text_trajectories = []
    eval_text_trajectories = []
    for personality, convos in raw_train.items():
        train_text_trajectories.extend(create_trajectories_from_conversations(convos, role, reward_mode))
    for personality, convos in raw_eval.items():
        eval_text_trajectories.extend(create_trajectories_from_conversations(convos, role, reward_mode))

    print(f"Initial dataset sizes: train: {len(train_text_trajectories)}, eval: {len(eval_text_trajectories)}")

    train_token_transitions = [text_transition_to_token_transition(TextTransition(text_trajectory, None), tokenizer) for text_trajectory in train_text_trajectories]
    eval_token_transitions = [text_transition_to_token_transition(TextTransition(text_trajectory, None), tokenizer) for text_trajectory in eval_text_trajectories]

    train_token_transitions = [token_transition for token_transition in train_token_transitions if token_transition.token_trajectory.tokens.shape[0] <= max_sequence_length]
    eval_token_transitions = [token_transition for token_transition in eval_token_transitions if token_transition.token_trajectory.tokens.shape[0] <= max_sequence_length]

    print(f"Final dataset sizes: train: {len(train_token_transitions)}, eval: {len(eval_token_transitions)}")

    train_data = ILQLDataset(
        token_transitions=train_token_transitions, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )

    eval_data = ILQLDataset(
        token_transitions=eval_token_transitions, 
        pad_token_id=tokenizer.pad_token_id, 
        max_len=max_sequence_length, 
    )
    print("finished loading dataset")

    if checkpoint_is_sharded and lm_checkpoint_path is not None:
        tail_checkpoint, head_checkpoint = os.path.split(lm_checkpoint_path.strip('/'))
        lm_checkpoint_path = os.path.join(tail_checkpoint, 'shard_%d' % (jax.process_index()), head_checkpoint)
    
    if checkpoint_is_sharded and value_checkpoint_path is not None:
        value_checkpoint_path = os.path.join(value_checkpoint_path, 'shard_%d' % (jax.process_index()))

    value_path_suffix = f'_{value_checkpoint_idx}' if value_checkpoint_idx is not None else ''

    print("loading model")    
    if model_name == "gpt2-xl" or model_name == "gpt2-medium":
        model, params, shard_rules = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=lm_checkpoint_path, 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=gradient_checkpoint, 
            seed=0, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
        value_base = load_gpt2_model(
            model_str=model_name, 
            from_pretrained=True, 
            checkpoint_path=lm_checkpoint_path if init_value_with_lm else (os.path.join(value_checkpoint_path, f'value_base{value_path_suffix}') if value_checkpoint_path is not None else None), 
            use_fp16=jax.default_backend() == 'tpu', 
            tokenizer=tokenizer, 
            gradient_checkpoint=True, 
            seed=2,
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
        if value_checkpoint_path is not None:
            target_value_base = load_gpt2_model(
                model_str=model_name, 
                from_pretrained=True, 
                checkpoint_path=lm_checkpoint_path if init_value_with_lm else (os.path.join(value_checkpoint_path, f'target_value_base{value_path_suffix}') if value_checkpoint_path is not None else None), 
                use_fp16=jax.default_backend() == 'tpu', 
                tokenizer=tokenizer, 
                gradient_checkpoint=True, 
                seed=2,
                gcloud_project=gcloud_project, 
                gcloud_token=gcloud_token, 
            )
            target_value_base_params = target_value_base.params
        else:
            target_value_base_params = jax.tree_util.tree_map(lambda x: x.clone(), value_base.params)
            
    else:
        raise NotImplementedError

    print("finished loading model")    
    
    model_config = model.config

    if use_adafactor:
        optim = optax.MultiSteps(
            optax.adafactor(
                learning_rate=lr, 
                multiply_by_parameter_scale=False, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
        optim_type = OptimType.AdaFactorMultiStep
    else:
        optim = optax.MultiSteps(
            optax.adamw(
                learning_rate=lr, 
                b1=0.9, 
                b2=0.999, 
                eps=1e-8, 
                weight_decay=weight_decay, 
            ), 
            every_k_schedule=grad_accum_steps, 
        )
        optim_type = OptimType.AdamWMultiStep

    # mesh definition
    if do_pjit:
        mesh_devices = np.array(jax.devices()).reshape(data_p_shape, model_p_shape)
        print('using mesh shape:', mesh_devices.shape)
        print('full mesh:', mesh_devices)
        mesh = Mesh(mesh_devices, ("dp", "mp"))
    else:
        mesh = contextlib.nullcontext()

    rng = jax.random.PRNGKey(seed)

    q_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=True, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    v_load_head_fn = partial(load_advanced_mlp4, inner_dim=model_config.hidden_size*4, 
                             dropout=model_config.resid_pdrop, add_state_term=False, shard_params=True, 
                             gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    
    rng, ilql_state_rng = jax.random.split(rng)
    ilql_state = load_ilql_default_train(
        value_base=value_base, 
        target_value_base_params=target_value_base_params, 
        q_load_head_fn=q_load_head_fn, 
        q_head_rng_keys=frozenset(['dropout']), 
        v_load_head_fn=v_load_head_fn, 
        v_head_rng_keys=frozenset(['dropout']), 
        rng=ilql_state_rng, 
        optim=optim, 
        optim_type=optim_type, 
        mesh=mesh, 
        do_pjit=do_pjit, 
        q1_checkpoint_path=os.path.join(value_checkpoint_path, f'q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        q2_checkpoint_path=os.path.join(value_checkpoint_path, f'q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        v_checkpoint_path=os.path.join(value_checkpoint_path, f'v_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q1_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q1_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
        target_q2_checkpoint_path=os.path.join(value_checkpoint_path, f'target_q2_head{value_path_suffix}.pkl') if value_checkpoint_path is not None else None, 
    )

    # shard params
    if do_pjit:
        params, param_spec = shard_params(partial(model.init_weights, input_shape=(1, 1)), 
                                                                params, shard_rules, mesh)
    else:
        param_spec = None

    base_lm_state = BaseModelState(
        model=model, 
        params=params, 
        param_spec=param_spec, 
    )

    loss_fn = partial(ilql_loss, gamma=0.99, tau=tau, cql_weight=cql_weight, detach_q1=False, detach_q2=False, detach_v=False, separate_cql=True)
    
    trainer = load_ilql_trainer(
        value_base_train_state=ilql_state.value_base_train_state, 
        q1_head_train_state=ilql_state.q1_head_train_state, 
        q2_head_train_state=ilql_state.q2_head_train_state, 
        v_head_train_state=ilql_state.v_head_train_state, 
        target_value_base_model_state=ilql_state.target_value_base_model_state, 
        target_q1_head_model_state=ilql_state.target_q1_head_model_state, 
        target_q2_head_model_state=ilql_state.target_q2_head_model_state, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=loss_fn, 
        polyak_alpha=0.005, 
        hard_update_every=None, 
        use_pre_ln_state=False, 
    )

    inference = load_ilql_inference(
        pi_beta_state=base_lm_state, 
        value_base_state=ilql_state.value_base_train_state.model_state, 
        q1_head_state=ilql_state.q1_head_train_state.model_state, 
        q2_head_state=ilql_state.q2_head_train_state.model_state, 
        v_head_state=ilql_state.v_head_train_state.model_state, 
        target_value_base_state=ilql_state.target_value_base_model_state, 
        target_q1_head_state=ilql_state.target_q1_head_model_state, 
        target_q2_head_state=ilql_state.target_q2_head_model_state, 
        tokenizer=tokenizer, 
        do_pjit=do_pjit, 
        loss_fn=loss_fn, 
        beta=inference_beta, 
        use_pre_ln_state=False, 
    )

    rng, evaluator_rng = jax.random.split(rng)

    def evaluator(inference: Inference):
        nonlocal evaluator_rng

        evaluator_rng, eval_loop_rng = jax.random.split(evaluator_rng)

        results = {}

        data_results = eval_loop(
            inference=inference, 
            dataset=eval_data, 
            rng=eval_loop_rng, 
            bsize=inference_bsize, 
            prefetch_batches=None, 
            eval_batches=eval_batches, 
        )
        results['data'] = data_results
        
        return data_results['loss'], results

    save_dir = None
    if exp_name is not None:
        save_dir = convert_path(os.path.join(output_path, exp_name, 'shard_%d' % (jax.process_index())))
        if (not save_dir.startswith('gcs://')) and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
        
        # copy training script to outputs as a cheap form of config logging
        with open(__file__, 'r') as f_local:
            with open(os.path.join(save_dir, 'config.py'), 'w') as f_save:
                f_save.write(f_local.read())
        with open(os.path.join(save_dir, 'input_args.pkl'), 'wb') as f:
            pkl.dump(input_args, f)
        
        # save info about mesh devices
        if do_pjit:
            with open(os.path.join(save_dir, 'system_mesh.pkl'), 'wb') as f:
                pkl.dump({'mesh': tree.map_structure(lambda x: {'id': int(x.id), 'process_index': int(x.process_index)}, mesh.devices.tolist()), 
                          'process_index': int(jax.process_index()), 'process_count': int(jax.process_count())}, f)
    
    n_datapoints = len(train_data)
    if log_every is None:
        log_every = int(n_datapoints / (train_bsize * num_logs_per_epoch))
    if eval_every is None:
        eval_every = int(n_datapoints / (train_bsize * num_evals_per_epoch))
    if save_every is None:
        save_every = int(n_datapoints / (train_bsize * num_saves_per_epoch))
    
    if save_best and not save_last:
        save_every = None

    rng, training_rng = jax.random.split(rng)
    with mesh:
        trainer, inference = train_loop(
            value_base_model=value_base.model, 
            target_value_base_model=value_base.model, 
            trainer=trainer, 
            inference=inference, 
            evaluator=evaluator, 
            dataset=train_data, 
            rng=training_rng, 
            save_dir=save_dir, 
            max_checkpoints=1 if save_last else None, 
            epochs=epochs, 
            max_steps=max_steps, 
            bsize=train_bsize, 
            prefetch_batches=None, 
            log_every=log_every, 
            eval_every=eval_every, 
            save_every=save_every,
            save_every_epochs=None,
            save_at_end=save_last, 
            save_best=save_best or save_best_also, 
            use_wandb=use_wandb, 
            wandb_project=wandb_project, 
            wandb_run_name=exp_name, 
            wandb_config=None, 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )

if __name__ == "__main__":
    tyro.cli(main)

from typing import Any, Callable, Dict, Optional, Tuple, Union, Hashable
from jaxtyping import PyTree
from jax.random import KeyArray
from collections import deque
import jax
from tqdm.auto import tqdm
from JaxSeq.utils import Dataset, dataloader, create_path, get_enabled_save_path
from JaxSeq.data import Seq2SeqDataset, Seq2SeqIterableDataset
from JaxSeq.models.base_interface import Train, Inference
from JaxSeq.logs import combine_logs, label_logs, log, pull_logs
import os
import wandb
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.bucket_manager import delete_with_bucket as delete
from JaxSeq.checkpointing import save_pytree
from JaxSeq.shard_model import get_sharding_from_model
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import pickle as pkl
from LLM_RL.algorithms.ilql.base_interface import ILQLTrain, ILQLInference
from LLM_RL.algorithms.value_rl_base.base_interface import ValueRLInference
import jax.numpy as jnp
import flax.linen as nn

def dump_state(
    base_model: FlaxPreTrainedModel, 
    q_head_model: nn.Module, 
    v_head_model: nn.Module, 
    base_train_state: TrainState, 
    target_base_params: Optional[PyTree], 
    q1_head_train_state: TrainState, 
    q2_head_train_state: TrainState, 
    v_head_train_state: TrainState, 
    q1_target_head_params: PyTree, 
    q2_target_head_params: PyTree, 
    save_dir: str, 
    save_train_state: bool, 
    enable_save: bool, 
    save_dtype: jnp.dtype, 
    **loop_state: Dict[Hashable, Any], 
):
    # dump loop_state
    with open(get_enabled_save_path(os.path.join(save_dir, 'loop_state.pkl'), enabled=enable_save), 'wb') as f:
        pkl.dump(loop_state, f)
    
    # save base
    if enable_save:
        create_path(os.path.join(save_dir, 'base'))
    # dump base_model config
    with open(get_enabled_save_path(os.path.join(save_dir, 'base', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(base_model.config.to_json_string())
    # dump train_state
    if save_train_state:
        save_pytree(
            tree=base_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'base', 'train_state.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(base_model, base_train_state), 
        )
    else:
        save_pytree(
            tree=base_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'base', 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(base_model, base_train_state.params), 
        )
    
    # save target_base
    if enable_save:
        create_path(os.path.join(save_dir, 'target_base'))
    if target_base_params is not None:
        # dump target_base_model config
        with open(get_enabled_save_path(os.path.join(save_dir, 'target_base', 'config.json'), enabled=enable_save), 'w') as f:
            f.write(base_model.config.to_json_string())
        # dump params
        save_pytree(
            tree=target_base_params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'target_base', 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(base_model, target_base_params), 
        )
    
    # save q1_head
    if enable_save:
        create_path(os.path.join(save_dir, 'q1_head'))
    # dump q1_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'q1_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(q_head_model.config.to_json_string())
    # dump train_state
    if save_train_state:
        save_pytree(
            tree=q1_head_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'q1_head', 'train_state.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(q_head_model, q1_head_train_state), 
        )
    else:
        save_pytree(
            tree=q1_head_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'q1_head', 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(q_head_model, q1_head_train_state.params), 
        )
    
    # save q2_head
    if enable_save:
        create_path(os.path.join(save_dir, 'q2_head'))
    # dump q2_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'q2_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(q_head_model.config.to_json_string())
    # dump train_state
    if save_train_state:
        save_pytree(
            tree=q2_head_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'q2_head', 'train_state.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(q_head_model, q2_head_train_state), 
        )
    else:
        save_pytree(
            tree=q2_head_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'q2_head', 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(q_head_model, q2_head_train_state.params), 
        )
    
    # save v_head
    if enable_save:
        create_path(os.path.join(save_dir, 'v_head'))
    # dump v_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'v_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(v_head_model.config.to_json_string())
    # dump train_state
    if save_train_state:
        save_pytree(
            tree=v_head_train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'v_head', 'train_state.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(v_head_model, v_head_train_state), 
        )
    else:
        save_pytree(
            tree=v_head_train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'v_head', 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(v_head_model, v_head_train_state.params), 
        )
    
    # save q1_target_head
    if enable_save:
        create_path(os.path.join(save_dir, 'q1_target_head'))
    # dump q1_target_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'q1_target_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(q_head_model.config.to_json_string())
    # dump params
    save_pytree(
        tree=q1_target_head_params, 
        path=get_enabled_save_path(os.path.join(save_dir, 'q1_target_head', 'params.msgpack'), enabled=enable_save), 
        dtype=save_dtype, 
        sharding=get_sharding_from_model(q_head_model, q1_target_head_params), 
    )

    # save q2_target_head
    if enable_save:
        create_path(os.path.join(save_dir, 'q2_target_head'))
    # dump q2_target_head config
    with open(get_enabled_save_path(os.path.join(save_dir, 'q2_target_head', 'config.json'), enabled=enable_save), 'w') as f:
        f.write(q_head_model.config.to_json_string())
    # dump params
    save_pytree(
        tree=q2_target_head_params, 
        path=get_enabled_save_path(os.path.join(save_dir, 'q2_target_head', 'params.msgpack'), enabled=enable_save), 
        dtype=save_dtype, 
        sharding=get_sharding_from_model(q_head_model, q2_target_head_params), 
    )


def eval_loss(
    inference: ILQLInference, 
    dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
    prng_key: Optional[KeyArray], 
    bsize: int, 
    eval_batches: Optional[int], 
) -> Dict[str, Any]:
    # setup evaluator loop state
    eval_logs = []

    # eval on batches
    prng_key, new_prng = jax.random.split(prng_key) if prng_key is not None else (None, None)
    d = dataloader(new_prng, dataset, bsize, truncate=True)
    for i, batch in tqdm(enumerate(d)):
        # conditionally terminate early
        if eval_batches is not None and i >= eval_batches:
            break

        # get eval logs
        _, info = inference.eval_loss(**batch)
        eval_logs.append(info)
    
    # gather and postproc eval logs
    eval_logs = pull_logs(combine_logs(eval_logs))
    return eval_logs

def train_loop(
    trainer: ILQLTrain, 
    inference: Union[ValueRLInference, ILQLInference], 
    evaluator: Optional[Callable[[Inference], Tuple[float, Dict[str, Any]]]], 
    dataset: Union[Seq2SeqDataset, Seq2SeqIterableDataset], 
    prng_key: KeyArray, 
    save_dir: Optional[str], 
    epochs: int, 
    max_steps: Optional[int], 
    bsize: int, 
    log_every: int, 
    eval_every_steps: Optional[int], 
    eval_every_epochs: Optional[int], 
    eval_at_beginning: bool, 
    eval_at_end: bool, 
    save_every_steps: Optional[int], 
    save_every_epochs: Optional[int], 
    save_at_beginning: bool, 
    save_at_end: bool, 
    save_best: bool, 
    max_checkpoints: Optional[int], 
    save_train_state: bool, 
    save_dtype: jnp.dtype, 
    use_wandb: bool, 
    wandb_project: Optional[str], 
    wandb_run_name: Optional[str], 
    wandb_config: Optional[Dict[str, Any]], 
    is_main_process: Optional[bool]=None, 
    **loop_state: Dict[Hashable, Any], 
) -> Tuple[Train, Inference]:
    assert (not use_wandb) or (use_wandb and wandb_project is not None)
    if is_main_process is None:
        is_main_process = jax.process_index() == 0
    
    # initalize wandb
    wandb_id = loop_state.get('wandb_id', None)
    if use_wandb and is_main_process:
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        wandb.init(
            project=wandb_project, 
            id=wandb_id, 
            name=wandb_run_name, 
            config=wandb_config, 
            reinit=True, 
            resume="allow", 
        )

    # initalize training loop state
    train_logs = []
    best_perf = loop_state.get('best_perf', float('inf'))
    saved_checkpoints = loop_state.get('saved_checkpoints', deque([]))
    step = 0
    steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None
    if 'steps_per_epoch' in loop_state:
        assert steps_per_epoch == loop_state['steps_per_epoch'], 'loop_state steps_per_epoch does not match dataset steps_per_epoch'
    epoch = -1

    def _save(
        name: str, 
        add_to_queue: bool, 
        **loop_state: Dict[Hashable, Any], 
    ):
        nonlocal saved_checkpoints
        print(f'saving checkpoint {name} ...')
        # conditionally delete old checkpoints
        if add_to_queue and is_main_process:
            if (max_checkpoints is not None) and (len(saved_checkpoints) >= max_checkpoints):
                delete(saved_checkpoints.popleft(), recursive=True)
        curr_save_dir = os.path.join(save_dir, name)
        if is_main_process:
            create_path(curr_save_dir)
        dump_state(
            base_model=trainer.base_model, 
            q_head_model=trainer.q_head_model, 
            v_head_model=trainer.v_head_model, 
            base_train_state=trainer.base_train_state, 
            target_base_params=trainer.target_base_params, 
            q1_head_train_state=trainer.q1_head_train_state, 
            q2_head_train_state=trainer.q2_head_train_state, 
            v_head_train_state=trainer.v_head_train_state, 
            q1_target_head_params=trainer.q1_target_head_params, 
            q2_target_head_params=trainer.q2_target_head_params, 
            save_dir=curr_save_dir, 
            save_train_state=save_train_state, 
            enable_save=is_main_process, 
            save_dtype=save_dtype, 
            **loop_state, 
        )
        if add_to_queue and is_main_process:
            saved_checkpoints.append(curr_save_dir)
        print('saved.')
    
    def _inference_update():
        nonlocal inference
        if isinstance(inference, ValueRLInference):
            inference = inference.replace(
                base_params=trainer.base_train_state.params, 
                q1_head_params=trainer.q1_head_train_state.params, 
                q2_head_params=trainer.q2_head_train_state.params, 
                v_head_params=trainer.v_head_train_state.params, 
            )
        elif isinstance(inference, ILQLInference):
            new_value_inference = inference.value_inference.replace(
                base_params=trainer.base_train_state.params, 
                q1_head_params=trainer.q1_head_train_state.params, 
                q2_head_params=trainer.q2_head_train_state.params, 
                v_head_params=trainer.v_head_train_state.params, 
            )
            new_target_value_inference = inference.target_value_inference.replace(
                base_params=trainer.target_base_params, 
                q1_head_params=trainer.q1_target_head_params, 
                q2_head_params=trainer.q2_target_head_params, 
            )
            inference = inference.replace(
                value_inference=new_value_inference, 
                target_value_inference=new_target_value_inference, 
            )
        else:
            raise NotImplementedError
    
    def _eval(
        **loop_state: Dict[Hashable, Any], 
    ):
        nonlocal best_perf
        # get eval logs
        _inference_update()
        eval_perf, eval_logs = evaluator(inference)

        # publish eval logs
        eval_logs = pull_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch}))
        log(eval_logs, use_wandb and is_main_process)

        # conditionally save best model and optimizer state
        if save_dir is not None and save_best and eval_perf < best_perf:
            print('new best model!')
            best_perf = eval_perf
            _save(
                name='best', 
                add_to_queue=False, 
                **{**loop_state, 'best_perf': best_perf}, 
            )
    
    # begin evaluation
    if evaluator is not None and eval_at_beginning:
        _eval(
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch,  
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )
    
    # save initial checkpoint
    if save_dir is not None and save_at_beginning:
        _save(
            name='initial', 
            add_to_queue=False, 
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )
    
    # begin training loop
    for epoch in tqdm(range(epochs)):
        prng_key, new_prng = jax.random.split(prng_key)
        d = dataloader(new_prng, dataset, bsize, truncate=True)
        for batch in tqdm(d, total=steps_per_epoch):
            
            # step model and get training logs
            prng_key, new_prng = jax.random.split(prng_key)
            if 'step' in loop_state and step < loop_state['step']:
                step += 1
                continue
            trainer, _, info = trainer.step(
                **batch, 
                prng_key=new_prng, 
                train=True, 
            )
            train_logs.append(info)
            
            # publish training logs and clear logs
            if (step + 1) % log_every == 0:
                logs = combine_logs(train_logs)
                logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                log(logs, use_wandb and is_main_process)
                train_logs = []
            
            # begin evaluation
            if evaluator is not None and eval_every_steps is not None and (step + 1) % eval_every_steps == 0:
                _eval(
                    # loop state metadata
                    best_perf=best_perf, 
                    step=step+1, 
                    epoch=epoch, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )
            
            # periodically save checkpoint
            if save_dir is not None and save_every_steps is not None and (step + 1) % save_every_steps == 0:
                _save(
                    name=f'step_{step+1}', 
                    add_to_queue=True, 
                    # loop state metadata
                    best_perf=best_perf, 
                    step=step+1, 
                    epoch=epoch, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )

            step += 1

            # conditionally terminate
            if max_steps is not None and step >= max_steps:
                break
        
        # begin evaluation
        if evaluator is not None and eval_every_epochs is not None and (epoch + 1) % eval_every_epochs == 0:
            _eval(
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        # periodically save checkpoint
        if save_dir is not None and save_every_epochs is not None and (epoch + 1) % save_every_epochs == 0:
            _save(
                name=f'epoch_{epoch}', 
                add_to_queue=True, 
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        # conditionally terminate
        if max_steps is not None and step >= max_steps:
            break
    
    # begin evaluation
    if evaluator is not None and eval_at_end:
        _eval(
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        _save(
            name='last', 
            add_to_queue=False, 
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )

    # stop wandb
    if use_wandb and is_main_process:
        wandb.finish()
    _inference_update()
    return trainer, inference

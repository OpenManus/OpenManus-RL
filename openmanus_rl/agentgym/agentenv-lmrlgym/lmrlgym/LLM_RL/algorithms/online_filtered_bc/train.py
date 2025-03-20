from typing import Any, Callable, Dict, Optional, Tuple, Union, Hashable
from jax.random import KeyArray
from collections import deque
import jax
from tqdm.auto import tqdm
from JaxSeq.utils import Dataset, dataloader, get_enabled_save_path, create_path
from JaxSeq.models.base_interface import TrainMask, InferenceMask
from JaxSeq.data import MaskDataset, MaskIterableDataset
from JaxSeq.logs import combine_logs, label_logs, log, pull_logs
import os
import wandb
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.bucket_manager import delete_with_bucket as delete
from JaxSeq.checkpointing import save_pytree
from flax.training.train_state import TrainState
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from JaxSeq.shard_model import get_sharding_from_model
import pickle as pkl
from LLM_RL.algorithms.ppo.base_interface import PPOPolicy
import os
import jax.numpy as jnp

def dump_state(
    model: FlaxPreTrainedModel, 
    train_state: TrainState, 
    save_dir: str, 
    save_train_state: bool, 
    enable_save: bool, 
    save_dtype: jnp.dtype, 
    **loop_state: Dict[Hashable, Any], 
):  
    # dump model config
    with open(get_enabled_save_path(os.path.join(save_dir, 'config.json'), enabled=enable_save), 'w') as f:
        f.write(model.config.to_json_string())
    # dump loop_state
    with open(get_enabled_save_path(os.path.join(save_dir, 'loop_state.pkl'), enabled=enable_save), 'wb') as f:
        pkl.dump(loop_state, f)
    # dump train_state
    if save_train_state:
        save_pytree(
            tree=train_state, 
            path=get_enabled_save_path(os.path.join(save_dir, 'train_state.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(model, train_state), 
        )
    else:
        save_pytree(
            tree=train_state.params, 
            path=get_enabled_save_path(os.path.join(save_dir, 'params.msgpack'), enabled=enable_save), 
            dtype=save_dtype, 
            sharding=get_sharding_from_model(model, train_state.params), 
        )

def train_loop(
    trainer: TrainMask,
    inference: InferenceMask,
    policy: PPOPolicy,
    load_dataset: Callable[[InferenceMask, PPOPolicy], Union[MaskDataset, MaskIterableDataset]],
    evaluator: Optional[Callable[[InferenceMask, PPOPolicy], Tuple[float, Dict[str, Any]]]],
    prng_key: KeyArray,
    save_dir: Optional[str],
    n_rounds: int,
    epochs: int,
    max_steps: Optional[int],
    bsize: int,
    log_every: int,
    eval_every_steps: Optional[int],
    eval_every_epochs: Optional[int],
    eval_every_rounds: Optional[int],
    eval_at_beginning: bool,
    eval_at_end: bool,
    save_every_steps: Optional[int],
    save_every_epochs: Optional[int],
    save_every_rounds: Optional[int],
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
) -> Tuple[TrainMask, InferenceMask, PPOPolicy]:
    print("entering training loop ...")
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
    epoch = -1
    round = -1

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
            model=trainer.model, 
            train_state=trainer.train_state, 
            save_dir=curr_save_dir, 
            save_train_state=save_train_state, 
            enable_save=is_main_process, 
            save_dtype=save_dtype, 
            **loop_state, 
        )
        if add_to_queue and is_main_process:
            saved_checkpoints.append(curr_save_dir)
        print('saved.')
    
    def _eval(
        **loop_state: Dict[Hashable, Any], 
    ):
        nonlocal best_perf
        nonlocal inference
        # get eval logs
        inference = inference.replace(params=trainer.train_state.params)
        policy.set_params(trainer.train_state.params)
        eval_perf, eval_logs = evaluator(inference, policy)

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
    
    # begin training loop
    for round in tqdm(range(n_rounds)):
        
        print(f'beginning round {round} ...')
        print(f"best performance: {best_perf}")

        # load dataset
        dataset = load_dataset(inference, policy)
        if dataset is None:
            continue

        steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None
        if 'steps_per_epoch' in loop_state:
            assert steps_per_epoch == loop_state['steps_per_epoch'], 'loop_state steps_per_epoch does not match dataset steps_per_epoch'

        # begin evaluation
        if evaluator is not None and eval_at_beginning:
            _eval(
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                round=round, 
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
                round=round, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        print("num epochs: ", epochs)
        for epoch in tqdm(range(epochs)):
            prng_key, new_prng = jax.random.split(prng_key)
            d = dataloader(new_prng, dataset, bsize, truncate=True)
            print("steps per epoch: ", steps_per_epoch)
            for batch in tqdm(d, total=steps_per_epoch):
                # step model and get training logs
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
                    logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch, 'round': round}))
                    log(logs, use_wandb and is_main_process)
                    train_logs = []
                
                # begin evaluation
                if evaluator is not None and eval_every_steps is not None and (step + 1) % eval_every_steps == 0:
                    _eval(
                        # loop state metadata
                        best_perf=best_perf, 
                        step=step+1, 
                        epoch=epoch, 
                        round=round, 
                        saved_checkpoints=saved_checkpoints, 
                        steps_per_epoch=steps_per_epoch, 
                        wandb_id=wandb_id, 
                    )
                
                # periodically save checkpoint
                if save_dir is not None and save_every_steps is not None and (step + 1) % save_every_steps == 0:
                    _save(
                        name='step_%d' % (step+1), 
                        add_to_queue=True, 
                        # loop state metadata
                        best_perf=best_perf, 
                        step=step+1, 
                        epoch=epoch, 
                        round=round, 
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
                    round=round, 
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
                    round=round, 
                    saved_checkpoints=saved_checkpoints, 
                    steps_per_epoch=steps_per_epoch, 
                    wandb_id=wandb_id, 
                )
            
            # conditionally terminate
            if max_steps is not None and step >= max_steps:
                break
        
        # begin evaluation
        if evaluator is not None and eval_every_rounds is not None and (round + 1) % eval_every_rounds == 0:
            _eval(
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                round=round, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        # periodically save checkpoint
        if save_dir is not None and save_every_rounds is not None and (round + 1) % save_every_rounds == 0:
            _save(
                name='round_%d' % (round), 
                add_to_queue=True, 
                # loop state metadata
                best_perf=best_perf, 
                step=step, 
                epoch=epoch, 
                round=round, 
                saved_checkpoints=saved_checkpoints, 
                steps_per_epoch=steps_per_epoch, 
                wandb_id=wandb_id, 
            )
        
        inference = inference.replace(params=trainer.train_state.params)
        policy.set_params(trainer.train_state.params)
    
    # begin evaluation
    if evaluator is not None and eval_at_end:
        _eval(
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            round=round, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        print("saving final checkpoint!")
        _save(
            name='last', 
            add_to_queue=False, 
            # loop state metadata
            best_perf=best_perf, 
            step=step, 
            epoch=epoch, 
            round=round, 
            saved_checkpoints=saved_checkpoints, 
            steps_per_epoch=steps_per_epoch, 
            wandb_id=wandb_id, 
        )

    # stop wandb
    if use_wandb and is_main_process:
        wandb.finish()
    
    inference = inference.replace(params=trainer.train_state.params)
    policy.set_params(trainer.train_state.params)
    return trainer, inference, policy

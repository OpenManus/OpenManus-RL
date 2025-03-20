import contextlib
from functools import partial
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from LLM_RL.algorithms.checkpoints import save_checkpoint_huggingface
from LLM_RL.algorithms.bc.interface import BCTrainer, BCInference
from jaxtyping import PyTree
from jax.random import KeyArray
from jax.experimental.maps import Mesh
from collections import deque
import jax
from tqdm.auto import tqdm
from JaxSeq.utils import Dataset, IterableDataset, block_sequences, BlockingStrategy, dataloader
from log_utils import combine_logs, label_logs, log, pull_logs
import os
from transformers.modeling_flax_utils import FlaxPreTrainedModel
import wandb
from LLM_RL.algorithms.bc.data import BCDataset, BCIterableDataset
from JaxSeq.bucket_manager import open_with_bucket as open

def eval_loop(
    inference: BCInference, 
    dataset: Union[BCDataset, BCIterableDataset], 
    rng: KeyArray, 
    bsize: int, 
    prefetch_batches: Optional[int], 
    eval_batches: Optional[int], 
):
    # setup evaluator loop state
    eval_logs = []

    # eval on batches
    rng, new_rng = jax.random.split(rng)
    d = dataloader(new_rng, dataset, bsize, prefetch_batches=prefetch_batches, truncate=True)
    for i, items in enumerate(d):
        
        # conditionally terminate early
        if eval_batches is not None and i >= eval_batches:
            break

        # get eval logs
        _, info = inference.eval_loss(items)
        eval_logs.append(info)
    
    # gather and postproc eval logs
    eval_logs = pull_logs(combine_logs(eval_logs))

    return eval_logs

def train_loop(
    model: FlaxPreTrainedModel, 
    trainer: BCTrainer, 
    inference: BCInference, 
    evaluator: Optional[Callable[[BCInference], Tuple[float, Dict[str, Any]]]], 
    dataset: Union[BCDataset, BCIterableDataset], 
    rng: KeyArray, 
    save_dir: Optional[str], 
    max_checkpoints: Optional[int], 
    epochs: int, 
    max_steps: Optional[int], 
    bsize: int, 
    prefetch_batches: Optional[int], 
    log_every: int, 
    eval_every: int, 
    save_every: Optional[int], 
    save_at_end: bool, 
    save_best: bool, 
    use_wandb: bool, 
    wandb_project: str, 
    wandb_run_name: Optional[str], 
    wandb_config: Optional[Dict[str, Any]], 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
) -> Tuple[BCTrainer, BCInference]:

    open = partial(open, gcloud_project=gcloud_project, gcloud_token=gcloud_token)
    
    # initalize wandb
    if use_wandb and jax.process_index() == 0:
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config, reinit=True)

    # initalize training loop state
    train_logs = []
    best_perf = float('inf')
    saved_checkpoints = deque([])
    step = 0
    steps_per_epoch = len(dataset) // bsize if isinstance(dataset, Dataset) else None

    # begin training loop
    for epoch in tqdm(range(epochs), disable=jax.process_index() > 0):
        rng, new_rng = jax.random.split(rng)
        d = dataloader(new_rng, dataset, bsize, prefetch_batches=prefetch_batches, truncate=True)
        for items in tqdm(d, total=steps_per_epoch, disable=jax.process_index() > 0):
            
            # step model and get training logs
            rng, new_rng = jax.random.split(rng)
            _, info, trainer = trainer.train_step(items, new_rng)
            train_logs.append(info)
            
            # publish training logs and clear logs
            if (step + 1) % log_every == 0:
                logs = combine_logs(train_logs)
                logs = pull_logs(label_logs(logs, 'train', {'step': step+1, 'epoch': epoch}))
                if jax.process_index() == 0:
                    log(logs, use_wandb)
                train_logs = []
            
            # begin evaluation
            if (evaluator is not None) and (step + 1) % eval_every == 0:

                # get eval logs
                inference = inference.update_params(trainer.params)
                eval_perf, eval_logs = evaluator(inference)

                # publish eval logs
                eval_logs = pull_logs(label_logs(eval_logs, 'eval', {'step': step+1, 'epoch': epoch}))
                if jax.process_index() == 0:
                    log(eval_logs, use_wandb)

                # conditionally save best model and optimizer state
                if save_dir is not None and save_best and eval_perf < best_perf:
                    print('new best model! Saving ...')
                    model_dir = os.path.join(save_dir, 'model')
                    save_checkpoint_huggingface(
                        model_dir, 
                        model=model, 
                        params=jax.device_get(trainer.params), 
                        gcloud_project=gcloud_project, 
                        gcloud_token=gcloud_token, 
                    )
                    print('saved.')
                    best_perf = eval_perf
            
            # periodically save checkpoint
            if save_dir is not None and save_every is not None and (step + 1) % save_every == 0:
                print('saving checkpoint...')

                # conditionally delete old checkpoints
                if (max_checkpoints is not None) and (len(saved_checkpoints) >= max_checkpoints):
                    os.system('rm -rf %s' % (saved_checkpoints.popleft()))

                model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
                save_checkpoint_huggingface(
                    model_dir, 
                    model=model, 
                    params=jax.device_get(trainer.params), 
                    gcloud_project=gcloud_project, 
                    gcloud_token=gcloud_token, 
                )
                saved_checkpoints.append(model_dir)
                print('saved.')
            
            # conditionally terminate
            if max_steps is not None and (step + 1) >= max_steps:
                break

            step += 1
        
        # conditionally terminate
        if max_steps is not None and (step + 1) >= max_steps:
            break
    
    # save final checkpoint
    if save_dir is not None and save_at_end:
        print('saving checkpoint...')
        model_dir = os.path.join(save_dir, 'model_%d' % (step+1))
        save_checkpoint_huggingface(
            model_dir, 
            model=model, 
            params=jax.device_get(trainer.params), 
            gcloud_project=gcloud_project, 
            gcloud_token=gcloud_token, 
        )
        print('saved.')

    # stop wandb
    if use_wandb and jax.process_index() == 0:
        wandb_run.finish()
    
    inference = inference.update_params(trainer.params)
    return trainer, inference

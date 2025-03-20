from typing import Optional, Dict, Any
import tyro
from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.utils import convert_path, load_mesh, setup_experiment_save, get_enabled_save_path, MapIterable, FileOpenIterable, BlockingStrategy, Padding, Truncation, create_path
import jax
import jax.numpy as jnp
from JaxSeq.utils import get_weight_decay_mask, jsonl_stream
from JaxSeq.generation_eval import generate_language
import os
import optax
from JaxSeq.models.gpt2.interface import GPT2Train, GPT2Inference
from JaxSeq.models.gpt2.load import load_train_state, ModelLoadMode
import pickle as pkl
from JaxSeq.data import Seq2SeqIterableDataset
from JaxSeq.train import eval_loss, train_loop
from jaxtyping import PyTree
import re
from JaxSeq.optimizers import GPT3Optimizer
from transformers.generation import GenerationConfig
import json
import numpy as np
from transformers import AutoTokenizer
from JaxSeq.bucket_manager import open_with_bucket as open

def main(
    model_load_mode: ModelLoadMode, 
    model_load_path: str, 
    train_data_path: str, 
    eval_data_path: str, 

    /,  # Mark the end of positional arguments.

    exp_name: Optional[str]=None, 
    outputs_path: Optional[str]=None, 

    data_mesh_shape: int=1, 
    fsdp_mesh_shape: int=1, 
    model_mesh_shape: int=-1, 

    use_wandb: bool=False, 
    wandb_project: Optional[str]=None, 

    epochs: int=1, 
    max_steps: Optional[int]=None, 

    weight_decay: float=0.001, 
    init_lr: float=0.0, 
    end_lr: float=0.0001, 
    lr: float=0.0001, 
    lr_warmup_steps: int=1000, 
    lr_decay_steps: int=1001, # no decay, so just needs to be > warmup steps
    bf16_momentum: bool=False, 
    multiply_by_parameter_scale: bool=True, 

    train_bsize: int=128, 
    grad_accum_steps: Optional[int]=1, 

    gradient_checkpointing: bool=False, 
    gradient_checkpointing_policy: str='nothing_saveable', 
    bf16_activations: bool=False, 

    max_input_length: int=256, 
    max_output_length: int=16, 

    log_every: int=256, 
    eval_every_steps: Optional[int]=256, 
    eval_every_epochs: Optional[int]=None, 
    eval_at_beginning: bool=False, 
    eval_at_end: bool=True, 
    
    save_every_steps: Optional[int]=None, 
    save_every_epochs: Optional[int]=None, 
    save_at_beginning: bool=False, 
    save_at_end: bool=False, 
    save_best: bool=True, 
    max_checkpoints: Optional[int]=None, 
    save_train_state: bool=True, 
    save_bf16: bool=True, 

    eval_loss_bsize: int=32, 
    eval_loss_batches: Optional[int]=None, 
    generation_bsize: int=4, 
    generation_batches: Optional[int]=None, 

    force_pad_embeddings: bool=False, 

    should_restore_loop_state: bool=False, 
):
    input_args = locals()
    print(input_args)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    mesh = load_mesh((data_mesh_shape, fsdp_mesh_shape, model_mesh_shape), ('dp', 'fsdp', 'mp'))
    is_main_process = jax.process_index() == 0
    print(f"Mesh: {mesh}")
    print(f"Is main process: {is_main_process}")
    
    
    
    

    train_data = Seq2SeqIterableDataset.from_str_iterable(
        MapIterable(
            lambda x: (tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), x['out_text']), 
            FileOpenIterable(convert_path(train_data_path), 'r', pipe=jsonl_stream), 
        ), 
        tokenizer=tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length, 
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length
        ), 
    )

    eval_data = Seq2SeqIterableDataset.from_str_iterable(
        MapIterable(
            lambda x: (tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), x['out_text']), 
            FileOpenIterable(convert_path(eval_data_path), 'r', pipe=jsonl_stream), 
        ), 
        tokenizer=tokenizer, 
        in_blocking_strategy=BlockingStrategy(
            padding=Padding.LEFT, 
            truncation=Truncation.LEFT, 
            max_length=max_input_length, 
        ), 
        out_blocking_strategy=BlockingStrategy(
            padding=Padding.RIGHT, 
            truncation=Truncation.RIGHT, 
            max_length=max_output_length, 
        ), 
    )

    def optim_getter(params: PyTree):
        mask = get_weight_decay_mask((
            "".join([r"\['ln_\d+'\]", re.escape("['bias']")]), 
            "".join([r"\['ln_\d+'\]", re.escape("['scale']")]), 
            re.escape("['ln_f']['bias']"), 
            re.escape("['ln_f']['scale']"), 
            "bias", 
        ))(params)

        optimizer_config = GPT3Optimizer(
            init_lr=init_lr, 
            end_lr=end_lr, 
            lr=lr, 
            lr_warmup_steps=lr_warmup_steps, 
            lr_decay_steps=lr_decay_steps, 
            weight_decay=weight_decay, 
            bf16_momentum=bf16_momentum, 
            multiply_by_parameter_scale=multiply_by_parameter_scale, 
        )

        optim, _ = optimizer_config.get_optim(mask)

        if grad_accum_steps is not None:
            return optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
        return optim

    model_prng_key = jax.random.PRNGKey(2)
    train_state, model = load_train_state(
        model_load_mode=model_load_mode, 
        model_load_path=convert_path(model_load_path) if model_load_mode != ModelLoadMode.HF else model_load_path, 
        model_dtype=jnp.bfloat16 if bf16_activations else jnp.float32, 
        optim_getter=optim_getter, 
        tokenizer=tokenizer, 
        mesh=mesh, 
        prng_key=model_prng_key, 
        force_pad_embeddings=force_pad_embeddings, 
        params_dtype=jnp.float32, 
    )
    model.config.gradient_checkpointing = gradient_checkpointing
    model.config.gradient_checkpointing_policy = gradient_checkpointing_policy

    loop_state = dict()
    if should_restore_loop_state and (model_load_mode in {ModelLoadMode.TRAIN_STATE, 
                                                          ModelLoadMode.TRAIN_STATE_PARAMS, 
                                                          ModelLoadMode.PARAMS}):
        with open(os.path.join(convert_path(model_load_path), 'loop_state.pkl'), 'rb') as f:
            loop_state = pkl.load(f)
    
    trainer = GPT2Train.load_train(
        train_state=train_state, 
        model=model, 
        tokenizer=tokenizer, 
    )

    inference = GPT2Inference.load_inference(
        params=train_state.params, 
        model=model, 
        tokenizer=tokenizer, 
    )

    save_dir, exp_name = setup_experiment_save(
        exp_name=exp_name, 
        outputs_path=convert_path(outputs_path), 
        input_args=input_args, 
        script__file__=__file__, 
        is_main_process=is_main_process, 
    )

    
    eval_prng = jax.random.PRNGKey(0)
    eval_round = 0
    def evaluator(inference: GPT2Inference):
        nonlocal eval_prng
        nonlocal eval_round

        loss_metrics = eval_loss(
            inference=inference, 
            dataset=eval_data, 
            prng_key=None, 
            bsize=eval_loss_bsize, 
            eval_batches=eval_loss_batches, 
        )

        generation_examples = []
        with open(convert_path(eval_data_path), 'r') as f:
            for item in jsonl_stream(f):
                if len(generation_examples) >= generation_bsize*generation_batches:
                    break
                generation_examples.append(item)
        
        eval_prng, new_prng = jax.random.split(eval_prng)
        generation_data = generate_language(
            inference=inference, 
            prompts=list(map(lambda x: tokenizer.bos_token+x['in_text'].removeprefix(tokenizer.bos_token), generation_examples)), 
            references=list(map(lambda x: x['stockfish_actions'], generation_examples)), 
            prng_key=new_prng, 
            bsize=generation_bsize, 
            generation_batches=generation_batches, 
            blocking_strategy=BlockingStrategy(
                padding=Padding.LEFT, 
                truncation=Truncation.LEFT, 
                max_length=max_input_length
            ), 
            generation_config=GenerationConfig(
                max_length=max_input_length+max_output_length, 
                do_sample=False, 
                num_beams=1, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.encode('\n')[0], 
                temperature=None, 
                top_k=None, 
                top_p=None, 
            ), 
        )

        for item in generation_data:
            generation = item['generation'].split('\n', 1)[1].replace(" ", "").strip()
            refs = list(map(lambda x: x.replace(" ", "").strip(), item['reference']))
            item['parsed_generation'] = generation
            item['refs'] = refs
            item['move_match'] = float(item['parsed_generation'] in item['refs'])

        if save_dir is not None:
            generations_save_dir = os.path.join(save_dir, 'generations', str(eval_round))
            if is_main_process:
                create_path(generations_save_dir)
            with open(get_enabled_save_path(
                os.path.join(generations_save_dir, 'generations.json'), 
                enabled=is_main_process, 
            ), 'w') as f:
                json.dump(generation_data, f)
        
        move_accuracy = np.mean(list(map(lambda x: x['move_match'], generation_data)))

        eval_round += 1

        return loss_metrics['loss'], {'loss_metrics': loss_metrics, 'move_accuracy': move_accuracy}
    
    train_prng = jax.random.PRNGKey(1)
    save_dtype = jnp.bfloat16 if save_bf16 else jnp.float32
    trainer, inference = train_loop(
        trainer=trainer, 
        inference=inference, 
        evaluator=evaluator, 
        dataset=train_data, 
        prng_key=train_prng, 
        save_dir=save_dir, 
        epochs=epochs, 
        max_steps=max_steps, 
        bsize=train_bsize, 
        log_every=log_every, 
        eval_every_steps=eval_every_steps, 
        eval_every_epochs=eval_every_epochs, 
        eval_at_beginning=eval_at_beginning, 
        eval_at_end=eval_at_end, 
        save_every_steps=save_every_steps, 
        save_every_epochs=save_every_epochs, 
        save_at_beginning=save_at_beginning, 
        save_at_end=save_at_end, 
        save_best=save_best, 
        max_checkpoints=max_checkpoints, 
        save_train_state=save_train_state, 
        save_dtype=save_dtype, 
        use_wandb=use_wandb, 
        wandb_project=wandb_project, 
        wandb_run_name=exp_name, 
        wandb_config=None, 
        is_main_process=is_main_process, 
        **loop_state, 
    )

if __name__ == "__main__":
    tyro.cli(main)

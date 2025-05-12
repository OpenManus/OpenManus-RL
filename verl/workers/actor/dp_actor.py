# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        #print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        #print(f"[DP_Actor._forward_micro_batch] Entered. use_remove_padding={self.use_remove_padding}, use_ulysses_sp={self.use_ulysses_sp}")
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            #print(f"[DP_Actor._forward_micro_batch] input_ids device: {input_ids.device}, shape: {input_ids.shape}")

            if self.use_remove_padding:
                #print(f"[DP_Actor._forward_micro_batch] Using remove_padding.")
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                #print(f"[DP_Actor._forward_micro_batch] input_ids_rmpad shape after unpad: {input_ids_rmpad.shape}")

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    #print(f"[DP_Actor._forward_micro_batch] Using Ulysses SP. SP size: {self.ulysses_sequence_parallel_size}")
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)
                    #print(f"[DP_Actor._forward_micro_batch] input_ids_rmpad shape after SP slice: {input_ids_rmpad.shape}, pad_size: {pad_size}")

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                #print(f"[DP_Actor._forward_micro_batch] logits_rmpad device: {logits_rmpad.device}, shape: {logits_rmpad.shape}")

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    #print(f"[DP_Actor._forward_micro_batch] Gathering outputs for SP.")
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                #print(f"[DP_Actor._forward_micro_batch] full_log_probs shape after pad_input: {full_log_probs.shape}")

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                #print(f"[DP_Actor._forward_micro_batch] Not using remove_padding.")
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #print(f"[DP_Actor._forward_micro_batch] logits device: {logits.device}, shape: {logits.shape}")
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #print(f"[DP_Actor._forward_micro_batch] log_probs shape: {log_probs.shape}, entropy shape: {entropy.shape}")
            
            #print(f"[DP_Actor._forward_micro_batch] Exiting.")
            return entropy, log_probs

    def _optimizer_step(self):
        #print(f"[DP_Actor._optimizer_step] Entered.")
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            #print(f"[DP_Actor._optimizer_step] Clipping grad norm for FSDP module.")
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            #print(f"[DP_Actor._optimizer_step] Clipping grad norm for standard module.")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        print(f"[DP_Actor._optimizer_step] Optimizer step done. Grad norm: {grad_norm}. Exiting.")
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        #print(f"[DP_Actor.compute_log_prob] Entered. Data meta_info: {data.meta_info}")
        # set to eval
        #print(f"[DP_Actor.compute_log_prob] Setting actor_module to eval mode.")
        self.actor_module.eval()
        #print(f"[DP_Actor.compute_log_prob] actor_module is in eval mode: {not self.actor_module.training}")

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        #print(f"[DP_Actor.compute_log_prob] Selected batch keys. input_ids shape: {batch['input_ids'].shape}, responses shape: {batch['responses'].shape}")

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            #print(f"[DP_Actor.compute_log_prob] Using dynamic batch size. max_token_len (incl. SP): {max_token_len}")
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            #print(f"[DP_Actor.compute_log_prob] Using fixed micro_batch_size: {micro_batch_size}")
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        #print(f"[DP_Actor.compute_log_prob] Starting micro-batch loop for {len(micro_batches)} micro-batches.")
        for i, micro_batch_data in enumerate(micro_batches):
            #print(f"[DP_Actor.compute_log_prob] Processing micro-batch {i+1}/{len(micro_batches)}. Device of input_ids: {micro_batch_data['input_ids'].device}")
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch_data, temperature=temperature)
            log_probs_lst.append(log_probs)
        #print(f"[DP_Actor.compute_log_prob] Micro-batch loop finished.")
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            #print(f"[DP_Actor.compute_log_prob] Reverting dynamic batch size ordering.")
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
        
        #print(f"[DP_Actor.compute_log_prob] Exiting. Final log_probs shape: {log_probs.shape}")
        return log_probs

    def update_policy(self, data: DataProto):
        #print(f"[DP_Actor.update_policy] Entered. Data meta_info: {data.meta_info}")
        # make sure we are in training mode
        #print(f"[DP_Actor.update_policy] Setting actor_module to train mode.")
        self.actor_module.train()
        #print(f"[DP_Actor.update_policy] actor_module is in train mode: {self.actor_module.training}")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        #print(f"[DP_Actor.update_policy] Grad accumulation: {self.gradient_accumulation}, Temp: {temperature}")

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        #print(f"[DP_Actor.update_policy] Selected batch keys for training. input_ids shape: {batch['input_ids'].shape}")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        #print(f"[DP_Actor.update_policy] Created dataloader for {len(dataloader)} mini-batches.")

        metrics = {}
        for batch_idx, mini_batch_data_container in enumerate(dataloader):
            #print(f"[DP_Actor.update_policy] Processing mini-batch {batch_idx+1}/{len(dataloader)}.")
            # split batch into micro_batches
            # mini_batch = mini_batch_data_container # already a TensorDict from split
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                #print(f"[DP_Actor.update_policy] Using dynamic micro-batch for mini-batch {batch_idx+1}. max_token_len: {max_token_len}")
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch_data_container, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                fixed_micro_batch_size = self.config.ppo_micro_batch_size
                #print(f"[DP_Actor.update_policy] Using fixed micro-batch size for mini-batch {batch_idx+1}: {fixed_micro_batch_size}")
                micro_batches = mini_batch_data_container.split(fixed_micro_batch_size)
            
            #print(f"[DP_Actor.update_policy] Mini-batch {batch_idx+1} split into {len(micro_batches)} micro-batches.")
            self.actor_optimizer.zero_grad()
            #print(f"[DP_Actor.update_policy] Optimizer zero_grad done for mini-batch {batch_idx+1}.")

            for i, micro_batch_data in enumerate(micro_batches):
                #print(f"[DP_Actor.update_policy] Forward/Backward for micro-batch {i+1}/{len(micro_batches)} of mini-batch {batch_idx+1}.")
                # Ensure data is on CUDA for the forward pass, FSDP handles sharding.
                # Verl's default behavior might keep it on CPU if offloading is used, then FSDP moves shards.
                # For direct model call, it must be on the device FSDP expects for its root module or where computation occurs.
                # If not using FSDP, or if FSDP is parameter-only offload, this explicit .cuda() is important.
                micro_batch_data_cuda = micro_batch_data.cuda() 
                #print(f"[DP_Actor.update_policy] Micro-batch {i+1} input_ids device: {micro_batch_data_cuda['input_ids'].device}")
                
                responses = micro_batch_data_cuda['responses']
                response_length = responses.size(1)
                attention_mask = micro_batch_data_cuda['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = micro_batch_data_cuda['loss_mask']
                old_log_prob = micro_batch_data_cuda['old_log_probs']
                advantages = micro_batch_data_cuda['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=micro_batch_data_cuda, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff
                #print(f"[DP_Actor.update_policy] Micro-batch {i+1} losses: pg_loss={pg_loss.item():.4f}, entropy_loss={entropy_loss.item():.4f}, policy_loss={policy_loss.item():.4f}")

                if self.config.use_kl_loss:
                    ref_log_prob = micro_batch_data_cuda['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef
                    #print(f"[DP_Actor.update_policy] Micro-batch {i+1} KL loss: {kl_loss.item():.4f}, updated policy_loss: {policy_loss.item():.4f}")

                loss = policy_loss / self.gradient_accumulation
                #print(f"[DP_Actor.update_policy] Micro-batch {i+1} final loss (for backward): {loss.item():.4f}")
                loss.backward()
                #print(f"[DP_Actor.update_policy] Micro-batch {i+1} backward pass done.")

                current_metrics_data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, current_metrics_data)

            grad_norm = self._optimizer_step()
            optimizer_step_metrics = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, optimizer_step_metrics)
            #print(f"[DP_Actor.update_policy] Optimizer step done for mini-batch {batch_idx+1}. Grad norm: {grad_norm.item():.4f}")

        self.actor_optimizer.zero_grad()
        #print(f"[DP_Actor.update_policy] Final optimizer zero_grad. Exiting. Metrics: {metrics}")
        return metrics

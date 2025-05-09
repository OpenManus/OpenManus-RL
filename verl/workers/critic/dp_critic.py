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
Implement a multiprocess PPOCritic
"""
import itertools
from typing import Iterable
from collections import defaultdict

import torch
import torch.distributed
from torch import nn, optim

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.critic import BasePPOCritic
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOCritic']


class DataParallelPPOCritic(BasePPOCritic):

    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        print(f'Critic use_remove_padding={self.use_remove_padding}')

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size

        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_micro_batch(self, micro_batch):
        print(f"[DP_Critic._forward_micro_batch] Entered. use_remove_padding={self.use_remove_padding}, use_ulysses_sp={self.ulysses_sequence_parallel_size > 1}")
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            print(f"[DP_Critic._forward_micro_batch] input_ids device: {input_ids.device}, shape: {input_ids.shape}")

            if self.use_remove_padding:
                print(f"[DP_Critic._forward_micro_batch] Using remove_padding.")
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                print(f"[DP_Critic._forward_micro_batch] input_ids_rmpad shape after unpad: {input_ids_rmpad.shape}")

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    print(f"[DP_Critic._forward_micro_batch] Using Ulysses SP. SP size: {self.ulysses_sequence_parallel_size}")
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    print(f"[DP_Critic._forward_micro_batch] input_ids_rmpad shape after SP slice: {input_ids_rmpad.shape}, pad_size: {pad_size}")

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)
                print(f"[DP_Critic._forward_micro_batch] values_rmpad shape after model: {values_rmpad.shape}")

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    print(f"[DP_Critic._forward_micro_batch] Gathering outputs for SP.")
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                print(f"[DP_Critic._forward_micro_batch] values shape after pad_input: {values.shape}")
                # Adjust slicing for critic: we need value for the state BEFORE each token in response
                values = values[:, -response_length - 1:-1]
                print(f"[DP_Critic._forward_micro_batch] values shape after slicing for response: {values.shape}")
            else:
                print(f"[DP_Critic._forward_micro_batch] Not using remove_padding.")
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
                print(f"[DP_Critic._forward_micro_batch] values device: {values.device}, shape: {values.shape}")
                # Adjust slicing for critic: we need value for the state BEFORE each token in response
                values = values[:, -response_length - 1:-1].squeeze(-1)
                print(f"[DP_Critic._forward_micro_batch] values shape after slicing for response: {values.shape}")
            
            print(f"[DP_Critic._forward_micro_batch] Exiting.")
            return values

    def _optimizer_step(self):
        print(f"[DP_Critic._optimizer_step] Entered.")
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            print(f"[DP_Critic._optimizer_step] Clipping grad norm for FSDP module.")
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            print(f"[DP_Critic._optimizer_step] Clipping grad norm for standard module.")
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        print(f"[DP_Critic._optimizer_step] Optimizer step done. Grad norm: {grad_norm}. Exiting.")
        return grad_norm

    def compute_values(self, data: DataProto):
        print(f"[DP_Critic.compute_values] Entered. Data meta_info: {data.meta_info}")
        # Assuming data.meta_info should contain 'micro_batch_size' and 'use_dynamic_bsz'
        # These should be set by the trainer before calling this method.
        if 'micro_batch_size' not in data.meta_info or 'use_dynamic_bsz' not in data.meta_info:
             print("[DP_Critic.compute_values] WARNING: 'micro_batch_size' or 'use_dynamic_bsz' missing from meta_info! This might cause errors.")
             # Assigning defaults here might mask the issue, but can prevent immediate crash
             micro_batch_size = data.meta_info.get('micro_batch_size', 1) # Default to 1 if missing
             use_dynamic_bsz = data.meta_info.get('use_dynamic_bsz', False)
             data.meta_info['micro_batch_size'] = micro_batch_size
             data.meta_info['use_dynamic_bsz'] = use_dynamic_bsz
        else:
            micro_batch_size = data.meta_info['micro_batch_size']
            use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        print(f"[DP_Critic.compute_values] Setting critic_module to eval mode.")
        self.critic_module.eval()
        print(f"[DP_Critic.compute_values] critic_module is in eval mode: {not self.critic_module.training}")
        
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        print(f"[DP_Critic.compute_values] Selected batch keys. input_ids shape: {batch['input_ids'].shape}, responses shape: {batch['responses'].shape}")

        # Verl's default behavior might send tensors on CPU if FSDP offload is used.
        # The forward pass needs data on the appropriate device.
        # Let's check the device before splitting.
        print(f"[DP_Critic.compute_values] Device BEFORE split: {batch['input_ids'].device}")

        # If tensors are not on CUDA, move them? FSDP might handle this automatically.
        # For now, assume FSDP handles device placement for forward pass.

        if use_dynamic_bsz:
            max_token_len = data.meta_info.get('max_token_len', 2048) * self.ulysses_sequence_parallel_size # Default if missing
            print(f"[DP_Critic.compute_values] Using dynamic batch size. max_token_len (incl. SP): {max_token_len}")
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            # micro_batch_size might be 0 if not set correctly by trainer
            if micro_batch_size <= 0:
                print(f"[DP_Critic.compute_values] ERROR: micro_batch_size is {micro_batch_size}. Cannot split batch. Check trainer config.")
                # Raise an error or return dummy data?
                micro_batch_size = 1
            print(f"[DP_Critic.compute_values] Using fixed micro_batch_size: {micro_batch_size}")
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        print(f"[DP_Critic.compute_values] Starting micro-batch loop for {len(micro_batches)} micro-batches.")
        for i, micro_batch_data in enumerate(micro_batches):
            print(f"[DP_Critic.compute_values] Processing micro-batch {i+1}/{len(micro_batches)}. Device of input_ids: {micro_batch_data['input_ids'].device}")
            # Move to GPU if needed for forward pass? Or assume FSDP handles?
            # micro_batch_data = micro_batch_data.cuda() # Tentative
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch_data)
            values_lst.append(values)
        print(f"[DP_Critic.compute_values] Micro-batch loop finished.")
        values = torch.concat(values_lst, dim=0)
        print(f"[DP_Critic.compute_values] Concatenated values shape: {values.shape}")

        # No need to multiply by mask here as _forward_micro_batch slices correctly
        # responses = data.batch['responses']
        # attention_mask = data.batch['attention_mask']
        # response_length = responses.size(1)
        # # values = values * attention_mask[:, -response_length - 1:-1] # Masking done internally or not needed if sliced?

        if use_dynamic_bsz:
            print(f"[DP_Critic.compute_values] Reverting dynamic batch size ordering.")
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        print(f"[DP_Critic.compute_values] Exiting. Final values shape: {values.shape}")
        # The function signature in BasePPOCritic implies it should return a DataProto
        # containing the values, not just the tensor.
        output = DataProto.from_dict(tensors={'values': values})
        return output

    def update_critic(self, data: DataProto):
        print(f"[DP_Critic.update_critic] Entered. Data meta_info: {data.meta_info}")
        # make sure we are in training mode
        print(f"[DP_Critic.update_critic] Setting critic_module to train mode.")
        self.critic_module.train()
        print(f"[DP_Critic.update_critic] critic_module is in train mode: {self.critic_module.training}")
        metrics = {}

        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        batch = data.select(batch_keys=select_keys).batch
        print(f"[DP_Critic.update_critic] Selected batch keys for training. input_ids shape: {batch['input_ids'].shape}")
        
        current_actual_batch_size = batch['input_ids'].shape[0]
        configured_critic_ppo_mini_batch_size = self.config.ppo_mini_batch_size

        dataloader = [] # Default to an empty dataloader

        if current_actual_batch_size == 0:
            print(f"[DP_Critic.update_critic] Current batch size is 0. Skipping PPO updates for this batch.")
        else:
            # Determine the effective mini-batch size for splitting.
            # It should not be larger than the current actual batch size.
            # It also shouldn't be less than 1 if there's data.
            effective_mini_batch_size_for_split = min(current_actual_batch_size, configured_critic_ppo_mini_batch_size)
            
            if effective_mini_batch_size_for_split < 1: # Should ideally not happen if current_actual_batch_size > 0
                print(f"[DP_Critic.update_critic] Warning: effective_mini_batch_size_for_split calculated as {effective_mini_batch_size_for_split} from current_actual_batch_size={current_actual_batch_size} and configured_critic_ppo_mini_batch_size={configured_critic_ppo_mini_batch_size}. Setting to 1.")
                effective_mini_batch_size_for_split = 1


            if effective_mini_batch_size_for_split < configured_critic_ppo_mini_batch_size:
                print(f"[DP_Critic.update_critic] Adjusting PPO mini-batch size for critic update from configured {configured_critic_ppo_mini_batch_size} to actual {effective_mini_batch_size_for_split} due to small input batch size ({current_actual_batch_size}).")
            
            try:
                dataloader = batch.split(effective_mini_batch_size_for_split)
            except Exception as e:
                print(f"[DP_Critic.update_critic] Error during batch.split with effective_mini_batch_size={effective_mini_batch_size_for_split}: {e}")
                print(f"[DP_Critic.update_critic] Batch keys: {batch.keys()}, input_ids shape: {batch['input_ids'].shape if 'input_ids' in batch else 'N/A'}")
                # Keep dataloader as empty list to skip epochs

        # Try to log the number of mini-batches.
        # Note: If dataloader is an iterator, len() might consume it or not be supported.
        # The original code used len(), implying it might be a list or has __len__.
        try:
            num_minibatches = len(dataloader)
            print(f"[DP_Critic.update_critic] Created dataloader for {num_minibatches} mini-batches.")
        except TypeError:
            # This happens if dataloader is an iterator without __len__
            # To get the length, one would need to convert to list, consuming it.
            # For now, we'll just note it's an iterator.
            print(f"[DP_Critic.update_critic] Created dataloader (iterator type, length not directly logged to avoid consumption).")


        metrics_to_avg = defaultdict(list)

        for batch_idx, mini_batch_data_container in enumerate(dataloader):
            print(f"[DP_Critic.update_critic] Processing mini-batch {batch_idx+1}/{len(dataloader)}.")
            # split batch into micro_batches
            # mini_batch = mini_batch_data_container
            # Get dynamic batching config from self.config (critic's own config), not data.meta_info for update loop
            use_dynamic_bsz_update = self.config.get('use_dynamic_bsz', False)
            if use_dynamic_bsz_update:
                # Use critic's configured max token length
                max_token_len = self.config.get('ppo_max_token_len_per_gpu', 2048) * self.ulysses_sequence_parallel_size
                print(f"[DP_Critic.update_critic] Using dynamic micro-batch for mini-batch {batch_idx+1}. max_token_len: {max_token_len}")
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch_data_container, max_token_len=max_token_len)
            else:
                fixed_micro_batch_size_update = self.config.ppo_micro_batch_size
                print(f"[DP_Critic.update_critic] Using fixed micro-batch size for mini-batch {batch_idx+1}: {fixed_micro_batch_size_update}")
                micro_batches = mini_batch_data_container.split(fixed_micro_batch_size_update)
            
            print(f"[DP_Critic.update_critic] Mini-batch {batch_idx+1} split into {len(micro_batches)} micro-batches.")
            self.critic_optimizer.zero_grad()
            print(f"[DP_Critic.update_critic] Optimizer zero_grad done for mini-batch {batch_idx+1}.")

            for i, micro_batch_data in enumerate(micro_batches):
                print(f"[DP_Critic.update_critic] Forward/Backward for micro-batch {i+1}/{len(micro_batches)} of mini-batch {batch_idx+1}.")
                # Assuming FSDP handles device placement, but check device.
                micro_batch_data_cuda = micro_batch_data.cuda()
                print(f"[DP_Critic.update_critic] Micro-batch {i+1} input_ids device: {micro_batch_data_cuda['input_ids'].device}")
                
                # input_ids = micro_batch_data_cuda['input_ids'] # Not directly needed for loss
                responses = micro_batch_data_cuda['responses']
                attention_mask = micro_batch_data_cuda['attention_mask']
                # position_ids = micro_batch_data_cuda['position_ids'] # Needed by _forward_micro_batch
                values = micro_batch_data_cuda['values']
                returns = micro_batch_data_cuda['returns']
                response_length = responses.size(1)

                # Mask for loss calculation corresponds to the response part where values/returns are defined
                eos_mask = attention_mask[:, -response_length - 1:-1]

                vpreds = self._forward_micro_batch(micro_batch_data_cuda)
                print(f"[DP_Critic.update_critic] Micro-batch {i+1} vpreds shape: {vpreds.shape}")

                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                
                # Normalize loss by gradient accumulation steps
                # Determine accumulation steps from config
                gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
                loss = vf_loss / gradient_accumulation
                print(f"[DP_Critic.update_critic] Micro-batch {i+1} losses: vf_loss={vf_loss.item():.4f}, clipfrac={vf_clipfrac.item():.4f}, final_loss={loss.item():.4f}")
                loss.backward()
                print(f"[DP_Critic.update_critic] Micro-batch {i+1} backward pass done.")

                loss_data_metrics = {
                    'critic/vf_loss': vf_loss.detach().item(),
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                }
                append_to_dict(metrics_to_avg, loss_data_metrics)

            grad_norm = self._optimizer_step()
            optimizer_step_metrics = {'critic/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, optimizer_step_metrics)
            print(f"[DP_Critic.update_critic] Optimizer step done for mini-batch {batch_idx+1}. Grad norm: {grad_norm.item():.4f}")
            
        self.critic_optimizer.zero_grad()
        print(f"[DP_Critic.update_critic] Final optimizer zero_grad. Exiting. Metrics: {metrics}")
        
        # BasePPOCritic expects a DataProto containing metrics
        return DataProto(meta_info={'metrics': metrics}) # Wrap metrics in DataProto

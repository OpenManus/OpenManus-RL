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
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                values = values[:, -response_length - 1:-1]
            else:
                output = self.critic_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            use_cache=False)  # prevent model thinks we are generating
                values = output.logits
                values = values[:, -response_length - 1:-1].squeeze(-1)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        self.critic_optimizer.step()
        return grad_norm

    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info['micro_batch_size']
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        
        # --- DEBUG: 记录原始数据设备 ---
        original_device = 'Unknown'
        if 'input_ids' in data.batch:
            original_device = data.batch['input_ids'].device
        print(f"[DP_Critic.compute_values] Start - Original data device: {original_device}")
        
        batch = data.select(batch_keys=select_keys).batch
        
        # --- DEBUG: 记录 select 后的设备 ---
        select_device = 'Unknown'
        if 'input_ids' in batch:
            select_device = batch['input_ids'].device
        print(f"[DP_Critic.compute_values] Device AFTER select: {select_device}")
        
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        # --- 修复: 先将数据移动到 CPU 再进行 split ---
        print(f"[DP_Critic.compute_values] Moving batch to CPU before split")
        batch_cpu = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            print(f"[DP_Critic.compute_values] Using dynamic batch size with max_token_len={max_token_len}")
            micro_batches, indices = rearrange_micro_batches(batch=batch_cpu, max_token_len=max_token_len)
        else:
            print(f"[DP_Critic.compute_values] Using fixed batch size with micro_batch_size={micro_batch_size}")
            micro_batches = batch_cpu.split(micro_batch_size)

        values_lst = []
        for mb_idx, micro_batch in enumerate(micro_batches):
            # --- DEBUG: 记录 micro_batch 设备 ---
            mb_device = 'Unknown'
            if 'input_ids' in micro_batch:
                mb_device = micro_batch['input_ids'].device
            print(f"[DP_Critic.compute_values] Micro-batch {mb_idx} device BEFORE potential .cuda(): {mb_device}")
            
            # 有条件地移动到 CUDA
            target_device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
            needs_move = False
            if 'input_ids' in micro_batch:
                if micro_batch['input_ids'].device != target_device:
                    needs_move = True
            
            if needs_move and torch.cuda.is_available():
                print(f"[DP_Critic.compute_values] Moving micro-batch {mb_idx} to {target_device}")
                micro_batch = micro_batch.to(target_device)
            elif not torch.cuda.is_available():
                print(f"[DP_Critic.compute_values] WARNING: CUDA not available. Staying on CPU.")
            else:
                print(f"[DP_Critic.compute_values] Micro-batch {mb_idx} already on target device. Skipping move.")
                
            # --- DEBUG: 记录移动后的设备 ---
            after_mb_device = 'Unknown'
            if 'input_ids' in micro_batch:
                after_mb_device = micro_batch['input_ids'].device
            print(f"[DP_Critic.compute_values] Micro-batch {mb_idx} device AFTER potential .cuda(): {after_mb_device}")
            
            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
                # --- DEBUG: 记录 values 设备 ---
                print(f"[DP_Critic.compute_values] Micro-batch {mb_idx} values device: {values.device}")
            values_lst.append(values)
            
        print(f"[DP_Critic.compute_values] Concatenating {len(values_lst)} micro-batches")
        values = torch.concat(values_lst, dim=0)
        print(f"[DP_Critic.compute_values] Concatenated values device: {values.device}")
        
        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.size(1)
        
        # 确保 values 和 attention_mask 在同一设备上
        if values.device != attention_mask.device:
            print(f"[DP_Critic.compute_values] Moving values from {values.device} to {attention_mask.device}")
            values = values.to(attention_mask.device)
            
        values = values * attention_mask[:, -response_length - 1:-1]

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=values.device)
            values = values[revert_indices]

        print(f"[DP_Critic.compute_values] Final values shape: {values.shape}, device: {values.device}")
        return values

    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        # --- DEBUG: Log initial input data device ---
        initial_device = 'Unknown'
        if 'input_ids' in data.batch:
            initial_device = data.batch['input_ids'].device
        print(f"[DP_Critic.update_critic] Start - Input data device: {initial_device}")
        # --- END DEBUG ---
        
        select_keys = ['input_ids', 'responses', 'attention_mask', 'position_ids', 'values', 'returns']
        # --- DEBUG: Log device before select ---
        print(f"[DP_Critic.update_critic] Device BEFORE select: {initial_device}")
        batch = data.select(batch_keys=select_keys).batch
        # --- DEBUG: Log device after select ---
        select_device = 'Unknown'
        if 'input_ids' in batch:
            select_device = batch['input_ids'].device
        print(f"[DP_Critic.update_critic] Device AFTER select: {select_device}")
        # --- END DEBUG ---
        
        # --- 关键修改: 先将数据移动到 CPU 再执行 split ---
        print(f"[DP_Critic.update_critic] Moving batch to CPU before split")
        batch_cpu = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        # --- DEBUG: Log device before split ---
        split_device = 'cpu'  # 现在应该确定是在 CPU 上
        print(f"[DP_Critic.update_critic] Device BEFORE split: {split_device}")
        dataloader = batch_cpu.split(self.config.ppo_mini_batch_size)
        # --- DEBUG: Log device after split (dataloader is iterator) ---
        print(f"[DP_Critic.update_critic] Dataloader created after split")
        # --- END DEBUG ---

        for batch_idx, data in enumerate(dataloader):
            # --- DEBUG: Log mini_batch device before micro-batch split ---
            mb_device = 'Unknown'
            if 'input_ids' in data:
                mb_device = data['input_ids'].device
            print(f"[DP_Critic.update_critic] Mini-batch {batch_idx} device: {mb_device}")
            # --- END DEBUG ---
            
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.critic_optimizer.zero_grad()

            for micro_batch_idx, data in enumerate(micro_batches):
                # --- DEBUG: Log device before potential .cuda() ---
                before_cuda_device = 'Unknown'
                target_device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
                needs_move = False
                if 'input_ids' in data:
                    before_cuda_device = data['input_ids'].device
                    if before_cuda_device != target_device:
                         needs_move = True
                print(f"[DP_Critic.update_critic] Micro-batch {batch_idx}-{micro_batch_idx} device BEFORE move check: {before_cuda_device}")
                # --- END DEBUG ---
                
                # Conditional .cuda() call
                if needs_move and torch.cuda.is_available():
                    print(f"[DP_Critic.update_critic] Moving micro-batch {batch_idx}-{micro_batch_idx} from {before_cuda_device} to {target_device}")
                    data = data.to(target_device) 
                elif not torch.cuda.is_available():
                     print(f"[DP_Critic.update_critic] WARNING: CUDA not available, cannot move micro-batch {batch_idx}-{micro_batch_idx}")
                else:
                    print(f"[DP_Critic.update_critic] Micro-batch {batch_idx}-{micro_batch_idx} already on target device {target_device}. Skipping move.")
                 
                # --- DEBUG: Log device after potential .cuda() ---
                after_cuda_device = 'Unknown'
                if 'input_ids' in data:
                    after_cuda_device = data['input_ids'].device
                print(f"[DP_Critic.update_critic] Micro-batch {batch_idx}-{micro_batch_idx} device AFTER move check: {after_cuda_device}")
                # --- END DEBUG ---
                
                input_ids = data['input_ids']
                responses = data['responses']
                attention_mask = data['attention_mask']
                position_ids = data['position_ids']
                values = data['values']
                returns = data['returns']
                response_length = responses.size(1)

                eos_mask = attention_mask[:, -response_length - 1:-1]

                # --- DEBUG: Log device before forward pass ---
                forward_input_device = 'Unknown'
                if 'input_ids' in data:
                     forward_input_device = data['input_ids'].device
                print(f"[DP_Critic.update_critic] Micro-batch {batch_idx}-{micro_batch_idx} device BEFORE forward pass: {forward_input_device}")
                # --- END DEBUG ---
                
                vpreds = self._forward_micro_batch(data)
                
                # --- DEBUG: Log vpreds device ---
                print(f"[DP_Critic.update_critic] Micro-batch {batch_idx}-{micro_batch_idx} vpreds device: {vpreds.device}")
                # --- END DEBUG ---

                # assert not torch.any(torch.isnan(vpreds)).item()

                vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                     values=values,
                                                                     returns=returns,
                                                                     eos_mask=eos_mask,
                                                                     cliprange_value=self.config.cliprange_value)
                loss = vf_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'critic/vf_loss': vf_loss.detach().item(),
                    'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                    'critic/vpred_mean': masked_mean(vpreds, eos_mask).detach().item(),
                }

                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'critic/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.critic_optimizer.zero_grad()
        return metrics

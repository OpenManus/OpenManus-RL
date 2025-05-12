#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree of Thought Rollout 验证脚本

这个脚本用于验证 Tree of Thought Agent 的 rollout 功能是否正确实现，
不进行实际训练。它将初始化所需组件，运行几个 ToT rollout 实例，
并详细打印探索过程和结果。
"""

import os
import sys
import torch
import argparse
import ray
import json
import time
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path

# 确保可以导入必要的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from enum import Enum
from openmanus_tot import OpenManusToTAgent, ToTConfig
from openmanus_rl.llm_agent.tensor_helper import TensorHelper, TensorConfig

# 导入VERL相关组件
from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tokenizer import hf_tokenizer
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.single_controller.base import Worker

class Role(Enum):
    ActorRollout = "actor_rollout"
    Critic = "critic"
    RefPolicy = "ref"
    RewardModel = "rm"

class ActorRolloutWorker:
    pass

class CriticWorker:
    pass

class RefPolicyWorker:
    pass

class RewardModelWorker:
    pass

def visualize_trajectory(trajectory, indent=0):
    """
    以人类可读的方式可视化轨迹
    """
    for i, step in enumerate(trajectory):
        role = step.get("from", "unknown")
        if role == "human":
            prefix = "🧑 Human:"
        elif role == "gpt":
            prefix = "🤖 Agent:"
        elif role == "env":
            prefix = "🌍 Env:"
        else:
            prefix = f"❓ {role}:"
        
        # 缩进并打印内容
        content = step.get("value", "").strip()
        # 如果内容很长，只显示前200个字符
        if len(content) > 200:
            content = content[:200] + "... [内容已截断]"
        
        print(f"{' ' * indent}{prefix} {content}")
        
        # 如果有奖励信息，显示它
        if "reward" in step:
            print(f"{' ' * (indent+4)}💰 Reward: {step['reward']}")

def validate_tot_agent(model_path: str, env_name: str = "webshop", env_server_base: str = "http://0.0.0.0", env_ports: List[int] = [36001], tot_strategy: str = "BFS", tot_beam_width: int = 3, tot_branches: int = 10, max_turns: int = 5, num_examples: int = 2, debug: bool = False):
    """
    验证 Tree of Thought Agent 实现

    Args:
        model_path: 模型路径
        env_name: 环境名称
        env_server_base: 环境服务器基地址
        env_ports: 环境服务器端口列表
        tot_strategy: ToT 搜索策略 ("BFS" 或 "DFS")
        tot_beam_width: ToT 束宽
        tot_branches: 最大分支数
        max_turns: 最大回合数
        num_examples: 要测试的示例数量
        debug: 是否启用额外的调试输出
    """    
    if not ray.is_initialized():
        ray.init(
            num_gpus=torch.cuda.device_count(),
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN',
                }
            }
        )

    print(f"🔍 验证 Tree of Thought Agent 实现...")
    print(f"📂 模型路径: {model_path}")
    print(f"🌍 环境: {env_name}")
    print(f"🔀 ToT 策略: {tot_strategy}")
    print(f"🔢 ToT 束宽: {tot_beam_width}")
    print(f"🌲 最大分支数: {tot_branches}")
    print(f"🔄 最大回合数: {max_turns}")

    # 确保模型路径存在
    local_model_path = copy_local_path_from_hdfs(model_path)
    
    print(f"🔄 加载分词器...")
    tokenizer = hf_tokenizer(local_model_path)
    
    role_worker_mapping = {
    Role.ActorRollout: ActorRolloutWorker.options(name="ActorRolloutWorker", max_restarts=3),
    Role.Critic: CriticWorker.options(name="CriticWorker", max_restarts=3),
    Role.RefPolicy: RefPolicyWorker.options(name="RefPolicyWorker", max_restarts=3),
    Role.RewardModel: RewardModelWorker.options(name="RewardModelWorker", max_restarts=3)
}

    class_dict = {
        "actor_rollout": RayClassWithInitArgs(
            cls=role_worker_mapping[Role.ActorRollout],
            config={
                "model": {"path": local_model_path, "enable_gradient_checkpointing": False},
                "rollout": {"name": "vllm", "temperature": 1.0, "top_k": 50, "top_p": 0.95, "gpu_memory_utilization": 0.7}
            },
            role="actor_rollout"
        ),
        "critic": RayClassWithInitArgs(
            cls=role_worker_mapping[Role.Critic],
            config={"critic_type": "simple", "learning_rate": 0.001, "use_gpu": True},
            role="critic"
        ),
        "ref": RayClassWithInitArgs(
            cls=role_worker_mapping[Role.RefPolicy],
            config={"model": {"path": local_model_path, "temperature": 0.7}},
            role="ref"
        ),
        "rm": RayClassWithInitArgs(
            cls=role_worker_mapping[Role.RewardModel],
            config={"reward_fn": None, "scale": 1.0},
            role="rm"
        )
    }


    print(f"🔄 初始化 RayWorkerGroup...")
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = RayWorkerGroup(resource_pool=RayResourcePool(process_on_nodes=[torch.cuda.device_count()], use_gpu=True, max_colocate_count=1, name_prefix='tot_validation'), ray_cls_with_init=worker_dict_cls)
    wg_spawn = wg_dict.spawn(prefix_set=['actor_rollout', 'critic', 'ref', 'rm'])
    actor_rollout_wg = wg_spawn['actor_rollout']
    for role, worker in wg_spawn.items():
        worker.init_model()
    
    print("✅ All workers initialized successfully.")


    # 创建 ToT 配置
    tot_config = ToTConfig(
        max_turns=max_turns,
        max_start_length=1024,
        max_prompt_length=2048,
        max_response_length=512,
        max_obs_length=1024,
        num_gpus=torch.cuda.device_count(),
        env_name=env_name,
        env_ports=env_ports,
        env_server_base=env_server_base,
        env_data_len=200,
        algorithm_config=None,
        tot_beam_width=tot_beam_width,
        tot_exploration_factor=2,
        tot_max_branches=tot_branches,
        tot_value_guidance=False,  # 暂时关闭,因为我们没有提供critic
        tot_temperature=1.0,
        tot_search_strategy=tot_strategy
    )
    
    # 创建 ToT Agent
    print(f"🔄 创建 Tree of Thought Agent...")
    tot_agent = OpenManusToTAgent(
        tokenizer=tokenizer,
        actor_rollout_wg=actor_rollout_wg,
        config=tot_config,
        is_validation=True,
        logger=None
    )
    
    # 创建简单的测试样例
    print(f"🔄 准备测试样例...")
    
    # 为特定环境准备提示
    if env_name == "webshop":
        prompts = [
            "I'm looking for a stylish black dress that I can wear to a formal dinner.",
            "Find me a comfortable pair of running shoes for men.",
        ]
    elif env_name == "alfworld":
        prompts = [
            "Put a clean mug in the coffee maker.",
            "Find a knife and place it on the counter.",
        ]
    else:
        prompts = [
            f"Help me complete this task in the {env_name} environment.",
            f"I need assistance with the {env_name} environment.",
        ]
    
    # 限制示例数量
    prompts = prompts[:num_examples]
    
    # 创建批次
    all_results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*80}")
        print(f"🧪 测试样例 {i+1}/{len(prompts)}: {prompt[:50]}...")
        print(f"{'='*80}\n")
        
        # 分词化提示
        tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        
        # 创建 DataProto
        gen_batch = DataProto.from_dict({
            'input_ids': tokenized_prompt['input_ids'],
            'attention_mask': tokenized_prompt['attention_mask'],
            'position_ids': torch.arange(tokenized_prompt['input_ids'].shape[1], dtype=torch.long).unsqueeze(0)
        })
        gen_batch.meta_info = {'idx': [i]}
        
        # 添加提示到元信息以便后续分析
        gen_batch.non_tensor_batch = {
            'prompt': prompt,
            'reward_model': [{}]
        }
        
        try:
            # 运行 ToT Rollout
            print(f"🔄 执行 Tree of Thought rollout...")
            start_time = time.time()
            
            # 执行 rollout
            rollout_output = tot_agent.run_llm_loop(gen_batch, output_dir="./tot_validation_output", global_steps=0)
            
            end_time = time.time()
            rollout_time = end_time - start_time
            
            print(f"✅ Rollout 完成! 耗时: {rollout_time:.2f} 秒")
            
            # 提取结果
            if hasattr(rollout_output, 'meta_info'):
                # 提取轨迹
                trajectories = rollout_output.meta_info.get('rollout_trajectory', [])
                if trajectories:
                    print(f"\n🌟 找到 {len(trajectories)} 个轨迹.")
                    
                    # 显示第一个轨迹
                    print(f"\n🌟 最佳轨迹:")
                    visualize_trajectory(trajectories[0])
                    
                    # 额外信息
                    tot_metrics = rollout_output.meta_info.get('metrics', {})
                    if tot_metrics:
                        for k, v in tot_metrics.items():
                            if k.startswith('tot_'):
                                print(f"📊 {k}: {v}")
                else:
                    print("❌ 没有找到轨迹在元信息中.")
                
                # 提取奖励
                if 'reward' in rollout_output.meta_info:
                    rewards = rollout_output.meta_info['reward']
                    print(f"💰 奖励: {rewards}")
                
                # 更多细节
                if debug:
                    print("\n🔍 调试信息:")
                    for key, value in rollout_output.meta_info.items():
                        if key not in ['rollout_trajectory']:
                            print(f"  - {key}: {value}")
            else:
                print("❌ rollout_output 没有 meta_info 属性.")
            
            # 收集结果
            all_results.append({
                'prompt': prompt,
                'success': True,
                'time': rollout_time,
                'num_trajectories': len(trajectories) if 'trajectories' in locals() else 0,
                'tot_metrics': tot_metrics if 'tot_metrics' in locals() and tot_metrics else {},
            })
            
        except Exception as e:
            import traceback
            print(f"❌ 执行 rollout 时出错: {e}")
            traceback.print_exc()
            
            all_results.append({
                'prompt': prompt,
                'success': False,
                'error': str(e),
            })
    
    # 显示汇总结果
    print(f"\n{'='*80}")
    print(f"📊 汇总结果:")
    print(f"{'='*80}")
    
    successes = sum(1 for r in all_results if r['success'])
    print(f"✅ 成功: {successes}/{len(all_results)}")
    
    if successes > 0:
        avg_time = sum(r['time'] for r in all_results if r['success']) / successes
        print(f"⏱️ 平均执行时间: {avg_time:.2f} 秒")
        
        avg_trajs = sum(r.get('num_trajectories', 0) for r in all_results if r['success']) / successes
        print(f"🌲 平均轨迹数: {avg_trajs:.2f}")
    
    # 保存结果到文件
    results_file = "tot_validation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            'config': {
                'model_path': model_path,
                'env_name': env_name,
                'tot_strategy': tot_strategy,
                'tot_beam_width': tot_beam_width,
                'tot_branches': tot_branches,
                'max_turns': max_turns,
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"📑 详细结果已保存到 {results_file}")
    
    # 清理
    if not debug:
        ray.shutdown()
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="验证 Tree of Thought Agent 实现")
    parser.add_argument("--model_path", type=str, default="./Qwen2.5-3B", help="模型路径")
    parser.add_argument("--env_name", type=str, default="webshop", help="环境名称")
    parser.add_argument("--env_server_base", type=str, default="http://0.0.0.0", help="环境服务器基地址")
    parser.add_argument("--env_port", type=int, default=36001, help="环境服务器端口")
    parser.add_argument("--tot_strategy", type=str, default="BFS", choices=["BFS", "DFS"], help="ToT 搜索策略")
    parser.add_argument("--tot_beam", type=int, default=3, help="ToT 束宽")
    parser.add_argument("--tot_branches", type=int, default=10, help="最大分支数")
    parser.add_argument("--max_turns", type=int, default=5, help="最大回合数")
    parser.add_argument("--num_examples", type=int, default=2, help="要测试的示例数量")
    parser.add_argument("--debug", action="store_true", help="启用额外的调试输出")
    
    args = parser.parse_args()
    
    # 展开路径中的波浪号
    if args.model_path.startswith("~"):
        args.model_path = os.path.expanduser(args.model_path)
    
    # 如果模型路径是相对路径且不以./开头，添加./
    if not args.model_path.startswith('/') and not args.model_path.startswith('./'):
        args.model_path = './' + args.model_path
    
    validate_tot_agent(
        model_path=args.model_path,
        env_name=args.env_name,
        env_server_base=args.env_server_base,
        env_ports=[args.env_port],
        tot_strategy=args.tot_strategy,
        tot_beam_width=args.tot_beam,
        tot_branches=args.tot_branches,
        max_turns=args.max_turns,
        num_examples=args.num_examples,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
# OpenManus-RL 开发指南

## 项目概述

OpenManus-RL 是一个用于训练大型语言模型（LLM）执行智能体任务的强化学习框架。该项目结合了两个主要仓库：

1. **AgentGym**：提供环境、奖励和智能体任务的评估工具
2. **Verl**：处理强化学习训练、轨迹生成方法和奖励计算

训练流程遵循管道架构：
1. 启动 AgentGym 对应环境服务
2. 初始化奖励管理器和轨迹生成工作组
3. 通过 OpenManus 智能体生成交互轨迹
4. 运行 PPO 或 GRPO 训练更新 LLM
5. 保存检查点并从步骤 3 重新开始

### 关键组件

- **数据表示**：使用 Hugging Face parquet 文件作为输入，使用 `DataProto` 进行内部数据表示
- **训练脚本**：`train_ppo.sh` 和 `train_grpo.sh` 编排整个训练过程
- **基础智能体**：在 `openmanus_rl/llm_agent/openmanus.py` 中实现，负责环境交互
- **奖励计算**：由 `verl/utils/reward_score/agentgym.py` 管理，计算来自 AgentGym 的累积奖励

## 核心组件

### Verl 框架

Verl 是底层强化学习框架，处理 RL 训练循环、轨迹生成机制和奖励计算。

#### DataProto

`DataProto` 是整个框架中使用的核心数据结构：

- 封装基于张量的数据（存储在 `.batch` 中）和非张量元数据（存储在 `.meta_info` 中）
- 提供批处理操作方法（切片、合并等）
- 处理设备放置和数据一致性

示例：
```python
data = DataProto.from_dict({
    'input_ids': input_tensor,
    'attention_mask': mask_tensor,
    'position_ids': position_tensor
})
data.meta_info['task_idx'] = task_indices
```

#### Ray Trainer

基于 Ray 的训练器（`verl/trainer/ppo/ray_trainer.py`）实现分布式 PPO 训练：

- `RayPPOTrainer`：管理整个训练过程，包括：
  - 环境初始化
  - 工作组协调
  - 优势函数计算
  - 策略更新
  - 验证

关键方法：
- `init_workers()`：初始化不同的工作角色
- `fit()`：主训练循环
- `_validate()`：对当前策略运行验证
- `_save_checkpoint()`：保存模型检查点

#### Rollout Worker Group

轨迹生成工作组从当前策略生成交互轨迹：

- 实现为基于 Ray 的工作组，可分布在多个节点上
- 处理生成、对数概率计算和策略更新
- 使用 VLLM 进行高效推理

#### 奖励计算

奖励计算由专用模块处理：

- `verl/utils/reward_score/agentgym.py`：专用于 AgentGym 环境
- 各种奖励模块支持不同类型的奖励（EM 分数、BLEU 等）
- `apply_kl_penalty()`：向原始奖励添加 KL 散度惩罚

### OpenManus Agent

OpenManus 智能体（`openmanus_rl/llm_agent/openmanus.py`）作为 RL 框架和环境之间的接口。

#### 关键类

- `AgentConfig`：智能体配置
- `OpenManusAgent`：处理环境交互的主智能体类

#### 核心方法

1. **run_llm_loop**
   ```python
   def run_llm_loop(self, gen_batch: DataProto, output_dir: str = None, global_steps: int = 0) -> DataProto:
   ```
   
   该方法编排一批环境的交互循环：
   - 接收初始提示作为输入
   - 使用线程池运行并行轨迹生成
   - 收集轨迹和奖励
   - 将结果格式化为用于训练的 DataProto
   - 如果启用，则处理可视化

2. **_run_single_rollout**
   ```python
   def _run_single_rollout(self, initial_prompt_ids: torch.Tensor, task_idx: int) -> Dict[str, Any]:
   ```
   
   执行单个环境交互：
   - 使用任务索引重置环境
   - 运行多回合的交互循环
   - 使用 LLM 生成响应
   - 处理响应并执行动作
   - 收集奖励和观察

3. **_convert_rollout_results_to_dataproto**
   ```python
   def _convert_rollout_results_to_dataproto(self, results: List[Dict], original_batch: DataProto) -> DataProto:
   ```
   
   将轨迹生成结果转换为可训练格式：
   - 将奖励与令牌序列对齐
   - 创建令牌级奖励张量
   - 连接和填充对话片段
   - 保留原始批次的元数据

### 训练脚本

训练脚本（`train_ppo.sh` 和 `train_grpo.sh`）编排整个过程：

1. 初始化环境：
   - 解析命令行参数
   - 为特定 AgentGym 环境创建专用 conda 环境
   - 启动环境服务器

2. 设置训练：
   - 配置数据路径和实验名称
   - 初始化日志记录
   - 设置超参数

3. 运行训练：
   - 使用适当的算法（PPO 或 GRPO）启动 Verl 训练器
   - 监控训练进度
   - 保存检查点

## 开发指南

### 添加新的奖励方法

要添加新的奖励方法（例如，过程奖励，结果奖励）：

1. **创建新的奖励模块**：
   ```bash
   # 在 reward_score 目录中创建新文件
   touch /home/kunlunz2/github_repos/OpenManus-RL/verl/utils/reward_score/my_reward.py
   ```

2. **实现奖励函数**：
   ```python
   # my_reward.py
   def compute_score(solution_str, ground_truth, **kwargs):
       # 你的奖励计算逻辑
       return reward_tensor
   ```

3. **在 `__init__.py` 中注册奖励**：
   ```python
   # 添加到 verl/utils/reward_score/__init__.py
   from .my_reward import compute_score as my_reward_compute_score
   
   SUPPORTED_REWARD_SCORE_FNS = {
       # ... 现有奖励
       'my_reward': my_reward_compute_score,
   }
   ```

4. **修改智能体以收集适当的信息**：
   - 更新 `OpenManusAgent._run_single_rollout` 以收集所需信息
   - 修改 `_convert_rollout_results_to_dataproto` 以正确格式化奖励

5. **在训练脚本中使用新奖励**：
   ```bash
   # 在 train_ppo.sh 或 train_grpo.sh 中添加：
   algorithm.reward_score_fn=my_reward
   ```

对于过程奖励（process reward）特别来说：
- 修改 `_run_single_rollout` 以跟踪中间步骤
- 更新奖励计算以考虑过程（采取的步骤）而不仅仅是结果

### 添加新环境

要集成来自 AgentGym 的新环境：

1. **准备环境包**：
   - 在 `openmanus_rl/agentgym/agentenv-<env_name>/` 中创建专用目录
   - 包含 `environment.yml` 用于 conda 环境规范
   - 添加 `setup.sh` 用于任何额外的设置步骤

2. **更新训练脚本**：
   - 在 `train_ppo.sh` 和 `train_grpo.sh` 的 case 语句中添加新环境：
   ```bash
   new_env)
       LAUNCH_CMD="new_env --host $AGENTGYM_HOST --port \$AGENTGYM_PORT"
       DEFAULT_PORT=XXXX
       ;;
   ```

3. **更新 OpenManus 智能体**：
   - 在 `_init_env_client` 中向 `ENV_TO_TASK_CLASS` 添加新环境
   ```python
   ENV_TO_TASK_CLASS = {
       # ... 现有环境
       "new_env": "NewEnvTask",
   }
   ```

4. **准备训练数据**：
   - 在 `data/<env_name>/` 中创建用于训练/验证的 parquet 文件
   - 在数据中定义适当的奖励模型

5. **测试集成**：
   ```bash
   ./train_ppo.sh --env_name new_env
   ```

### 扩展轨迹生成方法

要添加新的轨迹生成或动作模板方法：

1. **修改 OpenManus 智能体**：
   - 在 `postprocess_predictions` 中添加新的解析逻辑：
   ```python
   def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
       # 添加新的模板模式
       new_pattern = r'<new_action>(.*?)</new_action>'
       # ... 相应处理
   ```

2. **添加新的动作执行逻辑**：
   - 更新 `_run_single_rollout` 以处理新的动作类型
   - 修改动作执行逻辑以处理新模板

3. **更新提示模板**：
   - 修改 `create_react_prompt` 以包含新动作模板的指令
   ```python
   def create_react_prompt(task_description, tool_manager):
       # 添加新动作模板的指令
   ```

4. **配置智能体**：
   - 如果需要新参数，请更新 `AgentConfig`
   - 修改训练脚本以传递适当的配置

### 高级修改

对于更高级的修改，如更改训练算法或奖励结构：

1. **修改 PPO 算法**：
   - 更新 `verl/trainer/ppo/core_algos.py` 以进行算法更改
   - 在 `compute_advantage` 中修改优势计算

2. **更改轨迹生成工作组**：
   - 在 `verl/single_controller/ray/` 中创建新的工作组类
   - 在适当的工厂方法中注册工作组

3. **自定义数据处理**：
   - 修改 `_convert_rollout_results_to_dataproto` 以支持自定义数据格式
   - 如果需要，更新 `DataProto` 方法

## 结论

OpenManus-RL 提供了一个灵活的框架，用于在智能体环境中对 LLM 进行强化学习。通过理解核心组件并遵循此开发指南，您可以扩展框架以支持新的环境、奖励结构和动作模板。

有关 AgentGym 集成的更详细信息，请参阅 `/home/kunlunz2/github_repos/OpenManus-RL/openmanus_rl/agentgym/2nd_dev_docs` 中的文档。 
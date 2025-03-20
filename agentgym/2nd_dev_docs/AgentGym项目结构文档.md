
# AgentGym项目结构文档

## 1. 项目架构概览

AgentGym是一个框架，专为评估和开发基于大型语言模型(LLM)的通用智能体而设计。该框架特点是提供多样化的交互环境和任务，具有统一的格式（ReAct格式），并支持实时反馈和并发操作，易于扩展。

项目总体架构包括以下几个主要组件：

- **环境服务器**：不同环境部署在不同服务器或端口上，提供封装的HTTP服务
- **环境客户端**：接收环境服务器提供的服务并封装为用户可调用的函数
- **控制器**：连接智能体和环境，负责评估智能体、收集数据和训练智能体
- **训练器**：提供模型训练和演化的功能
- **数据集**：包括AgentTraj轨迹集和AgentEval基准测试集

## 2. 主要目录结构及其职责

```
AgentGym/
├── agentenv/                      # 核心包，包含主要功能实现
│   ├── agentenv/                  # 框架主要模块
│   │   ├── controller/            # 控制器模块，用于连接智能体与环境
│   │   ├── envs/                  # 环境客户端实现，每个文件对应一种环境
│   │   └── trainer/               # 训练器，包含行为克隆和智能体演化的实现
│   ├── examples/                  # 使用示例
│   └── utils/                     # 工具函数
├── agentenv-webshop/              # WebShop环境服务器实现
├── agentenv-webarena/             # WebArena环境服务器实现
├── agentenv-tool/                 # 工具使用相关环境服务器实现
├── agentenv-textcraft/            # TextCraft环境服务器实现
├── agentenv-sciworld/             # SciWorld环境服务器实现
├── agentenv-sqlgym/               # SQL相关环境服务器实现
├── agentenv-lmrlgym/              # LMRL环境服务器实现
├── agentenv-alfworld/             # ALFWorld环境服务器实现
├── agentenv-babyai/               # BabyAI环境服务器实现
├── docs/                          # 文档
└── assets/                        # 资源文件，如图片等
```

## 3. 关键模块的依赖关系图

```
                    ┌─────────────┐
                    │    Agent    │
                    └──────┬──────┘
                           │
                           ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Environment  │◄───┤  Controller  │────►    Trainer   │
│   Server     │    │              │    │              │
└──────┬───────┘    └──────────────┘    └──────────────┘
       │                    ▲                   ▲
       │                    │                   │
       ▼                    │                   │
┌──────────────┐            │                   │
│ Environment  │────────────┘                   │
│    Client    │                                │
└──────────────┘                                │
       ▲                                        │
       │                                        │
       │            ┌──────────────┐            │
       └────────────┤     Task     │────────────┘
                    └──────────────┘
```

## 4. 系统生命周期

### 4.1 任务执行生命周期

控制器在智能体、环境客户端和环境服务器之间的交互过程如下：

1. **初始化阶段**：
   - 创建`Agent`对象，包含模型和分词器
   - 创建特定的`Task`对象（如`BabyAITask`），该Task关联特定的`EnvClient`类（如`BabyAIEnvClient`）
   - `EnvClient`在初始化时会连接到环境服务器（envserver），并通过HTTP请求创建环境实例

2. **任务执行流程**：
   - `Evaluator`类（作为controller的具体实现）通过`eval`方法启动评估过程
   - `eval`方法调用`generate_experience`方法执行实际任务
   - `Task`的`generate_experience`方法处理整个交互循环：
     - 首先调用`client.reset(idx)`重置环境到指定任务索引
     - 然后通过`client.observe()`获取初始状态
     - 开始交互循环直到任务完成（done=True）或达到最大回合数：
       1. 使用LLM模型生成action（Agent的响应）
       2. 将action通过`client.step(action)`发送到环境
       3. 环境执行action并返回新状态、奖励和是否完成
       4. 收集对话历史和结果

3. **进入下一个任务**：
   - 在评估脚本中，通过循环遍历数据索引(`data_idxs`)来处理多个任务
   - 每个任务完成后，记录任务结果，然后进入下一个循环迭代（即下一个任务）
   - 通过每次使用不同的`data_idx`调用`evaluator.eval()`来切换到新任务
   - 每次调用`eval`都会重置环境到新索引，从而开始新任务

4. **通信机制**：
   - `EnvClient`通过HTTP API与环境服务器通信
   - 主要的通信方法有：
     - `_post`和`_get`方法发送HTTP请求
     - `reset`方法重置环境
     - `step`方法将Agent的行动发送到环境服务器并获取响应
     - `observe`方法获取当前状态的文本表示

5. **任务转换**：
   - 任务完成条件由环境服务器返回的`done`标志决定
   - 当`done=True`或达到最大回合数(`max_rounds`)时，当前任务结束
   - 评估脚本通过循环不同的`data_idx`实现任务之间的切换

### 4.2 交互流程图

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Evaluator   │    │     Task     │    │  EnvClient   │    │  EnvServer   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                    │                   │
       │     eval(idxs)    │                    │                   │
       ├──────────────────►│                    │                   │
       │                   │                    │                   │
       │                   │    reset(idx)      │                   │
       │                   ├───────────────────►│                   │
       │                   │                    │       reset       │
       │                   │                    ├──────────────────►│
       │                   │                    │                   │
       │                   │                    │◄──────────────────┤
       │                   │                    │                   │
       │                   │    observe()       │                   │
       │                   ├───────────────────►│                   │
       │                   │                    │    observation    │
       │                   │                    ├──────────────────►│
       │                   │                    │                   │
       │                   │                    │◄──────────────────┤
       │                   │◄───────────────────┤                   │
       │                   │                    │                   │
       │                   │  generate action   │                   │
       │                   ├─────┬─────────────┐│                   │
       │                   │     │ LLM模型生成  ││                   │
       │                   │     └─────────────┘│                   │
       │                   │                    │                   │
       │                   │    step(action)    │                   │
       │                   ├───────────────────►│                   │
       │                   │                    │       step        │
       │                   │                    ├──────────────────►│
       │                   │                    │                   │
       │                   │                    │◄──────────────────┤
       │                   │◄───────────────────┤                   │
       │                   │                    │                   │
       │                   │                    │                   │
       │   返回执行结果     │                    │                   │
       │◄──────────────────┤                    │                   │
       │                   │                    │                   │
       │  进入下一个任务    │                    │                   │
       ├──────────────────►│                    │                   │
       │                   │                    │                   │
       └───────────────────┴────────────────────┴───────────────────┘
```

## 5. 核心类和接口的功能说明

### 5.1 控制器模块 (`controller/`)

控制器模块是整个AgentGym框架的核心组件，充当智能体与环境之间的桥梁，负责协调它们之间的交互，管理数据流动，并提供评估和训练的基础设施。该模块由以下几个关键文件组成：

- **agent.py**: 定义了智能体的基本接口
- **env.py**: 提供环境客户端的抽象接口
- **task.py**: 实现任务的基本框架和会话处理
- **utils.py**: 包含控制器、评估器和训练器的实现
- **__init__.py**: 导出模块的主要组件

此模块的核心类及其功能如下：

- **Agent**: 基本智能体类，封装了大型语言模型(LLM)和对应的分词器
  ```python
  class Agent:
      def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
          self.model = model
          self.tokenizer = tokenizer
  ```

- **BaseEnvClient**: 环境客户端抽象基类，定义了环境客户端的标准接口，所有具体环境客户端都需要实现这些接口
  ```python
  class BaseEnvClient(metaclass=ABCMeta):
      conversation_start = ()
      
      @abstractmethod
      def __len__(self) -> int: 
          """返回环境的总大小"""
          
      @abstractmethod
      def observe(self) -> str:
          """解析环境服务器响应并提供文本消息给LLM"""
          
      @abstractmethod
      def step(self, action) -> StepOutput:
          """解析模型输出的动作并调用环境服务器"""
          
      @abstractmethod
      def reset(self, idx: int) -> None:
          """重置环境"""
  ```

- **BaseTask**: 任务基类，连接环境客户端和控制器，管理环境客户端实例，提供会话标记化功能，实现经验生成功能
  ```python
  class BaseTask:
      env_client_cls: Callable
      env_name: str
      
      def __init__(self, client_args: Mapping[str, Any], n_clients: int = 1) -> None:
          """初始化Task对象"""
          
      def _tokenize_conversation(self, conversation, tokenizer) -> TokenizedConversationOutput:
          """将对话转换为模型可处理的标记化形式"""
          
      def generate_experience(self, model, tokenizer, idxs, generation_config, max_rounds) -> list[ExperienceOutput]:
          """让智能体在环境中执行并收集经验轨迹"""
  ```

- **BaseAgentEnvController**: 控制器基类，管理Agent和多个Task的交互，提供生成经验的统一接口
  ```python
  class BaseAgentEnvController:
      def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
          """初始化控制器"""
          
      def generate_experience(self, idxs, generation_config, max_rounds) -> list[ExperienceOutput]:
          """生成智能体在环境中的交互经验"""
  ```

- **Evaluator**: 评估器类，继承自BaseAgentEnvController，用于评估智能体在任务上的表现，计算平均奖励和成功率等指标
  ```python
  class Evaluator(BaseAgentEnvController):
      def eval(self, generation_config, max_rounds, idxs) -> EvaluationOutput:
          """评估智能体在给定任务上的表现，返回评估结果"""
  ```

- **BaseTrainer**: 训练器基类，也继承自BaseAgentEnvController，提供训练、评估和保存模型的接口
  ```python
  class BaseTrainer(BaseAgentEnvController):
      def train(self): 
          """训练方法"""
          
      def eval(self, generation_config, max_rounds, idxs) -> EvaluationOutput:
          """评估方法"""
          
      def save_model(self):
          """保存模型"""
  ```

控制器模块通过统一的接口处理不同环境和任务，使开发者可以专注于智能体算法的开发，而不必担心不同环境的细节差异。它是框架的中心组件，负责协调整个系统的运行和数据流动。

### 5.2 环境模块 (`envs/`)

每个环境都有一个对应的客户端类（如WebshopEnvClient）和任务类（如WebshopTask），分别继承自BaseEnvClient和BaseTask。环境包括：

1. WebShop
2. WebArena
3. MAZE
4. Wordle
5. ALFWorld
6. SciWorld
7. BabyAI
8. TextCraft
9. Weather
10. Movie
11. Academia
12. Sheet
13. TODOList
14. BIRD (SQL)

### 5.3 训练器模块 (`trainer/`)

- **BaseTrainer**: 训练器基类
  ```python
  class BaseTrainer(BaseAgentEnvController):
      def train(self): 
          """训练方法"""
          
      def eval(self, generation_config, max_rounds, idxs) -> EvaluationOutput:
          """评估方法"""
          
      def save_model(self):
          """保存模型"""
  ```

- **AgentEvolTrainer**: 智能体演化训练器
  ```python
  class AgentEvolTrainer(BaseTrainer):
      def train(self):
          """训练方法实现"""
          
      def evol(self):
          """演化方法实现"""
  ```

- **BCTrainer**: 行为克隆训练器

## 6. 数据流向图

```
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│   输入指令   │───────►  │    智能体    │───────►  │  环境客户端  │
└─────────────┘          └─────────────┘          └─────────────┘
                              ▲  │                       │
                              │  │                       │
                              │  ▼                       ▼
┌─────────────┐          ┌─────────────┐          ┌─────────────┐
│    反馈     │◄─────────┤    控制器    │◄─────────┤  环境服务器  │
└─────────────┘          └─────────────┘          └─────────────┘
      ▲                         │
      │                         │
      │                         ▼
┌─────────────┐          ┌─────────────┐
│  评估结果   │◄─────────┤    评估器    │
└─────────────┘          └─────────────┘
```

## 7. API接口清单

### 7.1 环境服务器HTTP接口

1. `/createEnv`: 创建环境
2. `/observation`: 获取当前环境观察
3. `/available_actions`: 获取当前可用动作
4. `/step`: 执行动作
5. `/reset`: 重置环境

### 7.2 主要类方法接口

1. `Agent.__init__(model, tokenizer)`: 初始化智能体
2. `BaseEnvClient.observe()`: 观察环境
3. `BaseEnvClient.step(action)`: 执行动作
4. `BaseEnvClient.reset(idx)`: 重置环境
5. `BaseTask.generate_experience(model, tokenizer, idxs, generation_config, max_rounds)`: 生成经验
6. `Evaluator.eval(generation_config, max_rounds, idxs)`: 评估智能体
7. `BaseTrainer.train()`: 训练智能体
8. `AgentEvolTrainer.evol()`: 演化智能体

## 8. 常见的代码模式和约定

1. **环境客户端实现模式**:
   - 每个环境客户端类继承`BaseEnvClient`基类
   - 实现`__len__`, `observe`, `step`, `reset`方法
   - 定义`conversation_start`作为初始对话

2. **任务实现模式**:
   - 每个任务类继承`BaseTask`基类
   - 指定对应的`env_client_cls`和`env_name`
   - 可选地覆盖`_tokenize_conversation_one`方法以适应特定任务

3. **训练和评估流程**:
   - 使用`Agent`类封装模型和分词器
   - 使用`Task`类连接环境客户端
   - 使用`Evaluator`或`Trainer`类管理评估和训练

4. **数据格式约定**:
   - 使用ReAct格式组织对话和行动
   - 交互记录使用`ConversationMessage`

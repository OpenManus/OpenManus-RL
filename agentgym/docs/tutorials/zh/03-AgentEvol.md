# 自我进化

AgentGYM 提出了 AgentEvol 算法，使模型能自我进化。

## 启动环境服务器

自我进化需要模型在环境中探索，请参考 [01-evaluation](docs/tutorials/zh/01-evaluation.md) 及各个环境服务器文件夹中的 `README.md` 启动对应的环境服务器。如果启动多个环境服务器，请注意端口号的选择。

## 自我进化

`agentenv/examples/agentevol` 给出了自我进化算法的参考实现。请参考 `examples/agentevol/train_agentevol_multi_task.sh` 启动训练。

# Continue Debugger 功能说明

## 概述

Continue Debugger 是对现有调试器的增强，旨在解决传统debugger的局限性。传统debugger只在错误点进行反馈并重新开始，但agent在后续步骤中可能会重复犯类似的错误。Continue Debugger通过维护累积的指导建议，在每一步都提醒agent避免相关错误。

## 主要特性

### 1. 累积指导建议 (Cumulative Guidance)
- 每次失败后，debugger不仅分析错误，还生成通用的follow-up instruction
- 这些指导建议会在后续的所有步骤中被注入到observation中
- 形成逐步积累的经验指导，帮助agent避免重复错误

### 2. 持续学习机制
- 每次重试时，debugger会：
  1. 回顾之前的指导建议
  2. 分析当前失败与之前模式的关系
  3. 生成新的、更针对性的指导建议
  4. 将新指导添加到累积列表中

### 3. 智能指导注入
- 在每个action步骤前，将累积的指导建议注入到observation中
- 格式化为清晰的指导列表，便于agent理解和应用
- 避免重复相同的指导建议

## 使用方法

### 基本用法
```bash
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --enable_debugger \
    --debugger_type continue \
    --debugger_model gpt-4.1 \
    --max_try 5
```

### 参数说明
- `--debugger_type continue`: 启用continue debugger模式
- `--debugger_model`: 用于分析和生成指导建议的模型
- `--debugger_temperature`: debugger模型的温度参数
- `--max_try`: 最大重试次数（每次重试都会积累新的指导）

### 测试脚本
```bash
# 运行预配置的测试
./scripts/rollout/test_continue_debugger.sh
```

## 输出文件

Continue Debugger会生成以下额外的输出：

### 1. 调试分析文件
每次重试都会生成`debug_analysis_retry_N.json`文件，包含：
- 错误分析结果
- 新生成的follow_up_instruction
- 当前累积的所有指导建议

### 2. 任务摘要
`task_summary.json`中会包含：
- 每次重试的成功/失败状态
- 累积的指导建议历史
- 最终的任务完成结果

## 示例工作流程

1. **第一次尝试**: Agent正常执行任务
2. **失败分析**: Continue debugger分析失败原因，生成first follow-up instruction
3. **第二次尝试**: 在每个observation中注入第一条指导建议
4. **再次失败**: 分析新失败，考虑之前的指导，生成第二条指导建议
5. **第三次尝试**: 在每个observation中注入两条累积的指导建议
6. **持续优化**: 直到成功或达到最大重试次数

## 与传统Debugger的区别

| 特性 | 传统Debugger (naive/advanced) | Continue Debugger |
|------|-------------------------------|-------------------|
| 错误定位 | ✓ | ✓ |
| 即时反馈 | ✓ | ✓ |
| 累积学习 | ✗ | ✓ |
| 持续指导 | ✗ | ✓ |
| 防止重复错误 | 部分 | ✓ |

## 适用场景

Continue Debugger特别适用于：
- 多步骤复杂任务
- 容易重复犯错的场景
- 需要逐步优化策略的任务
- 长期学习和改进的场景

## 注意事项

1. **计算开销**: 每次重试都需要额外的LLM调用来生成follow-up instruction
2. **提示长度**: 随着重试次数增加，注入的指导建议会增长，可能影响提示长度
3. **模型兼容性**: 需要支持JSON格式输出的模型来生成结构化的指导建议

## 技术实现

Continue Debugger基于以下核心组件：
- `ContinuousInstructionManager`: 管理累积的指导建议
- 增强的`LLMDebugger.analyze_trajectory()`: 支持生成follow-up instructions
- 修改的observation注入逻辑: 在每步都包含累积指导
- 扩展的参数系统: 支持continue模式配置

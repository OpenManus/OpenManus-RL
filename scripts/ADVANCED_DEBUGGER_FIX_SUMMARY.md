# Advanced Debugger 修复总结

## 问题描述
Advanced Debugger 的分析结果不正确，返回的都是 fallback 结果而不是真正的 API 调用结果。主要问题包括：
- "reason": "Extracted from malformed JSON response"
- "confidence": 0.3
- 没有正确调用 Advanced Debugger API

## 修复内容

### 1. 移除所有 Fallback 逻辑
- 删除了 `_convert_result` 方法中的所有默认值和 fallback 处理
- 创建了新的 `_convert_api_result` 方法，强制要求所有必需字段都存在
- 如果 API 返回的数据不完整，现在会抛出明确的异常而不是返回默认值

### 2. 修正 API 调用格式
- 创建了 `_build_trajectory_json` 方法来构建正确的 API 输入格式
- 确保 `trajectory_json` 包含所有必需的字段：
  - `metadata`: 包含环境信息和任务状态
  - `messages`: 完整的对话历史
  - `trajectory`: 轨迹步骤
  - `environment`: 环境类型
  - `task_success`: 任务是否成功

### 3. 改进错误处理和日志
- 添加了详细的日志记录，包括：
  - API 调用前的数据验证
  - API 响应的结构和内容
  - 关键错误的详细信息
- 添加了异常堆栈跟踪以便调试
- 验证 trajectory_json 结构的完整性

### 4. 修复代码 Bug
- 修复了第 1714 行的 `first` 未定义错误
- 更新了 ToT/DFSDT 策略的摘要文件生成逻辑

## 关键更改

### AdvancedDebugger.analyze_trajectory
```python
# 构建正确的 trajectory_json 格式
trajectory_json = self._build_trajectory_json(trajectory, env_type, chat_history, metadata)

# 调用 API 并处理结果
result = self._run_async(self.detector.analyze_trajectory(trajectory_json))

# 转换结果到期望的格式（无 fallback）
converted = self._convert_api_result(result, trajectory, env_type)
```

### _convert_api_result 方法
- 严格验证所有必需字段
- 不提供任何默认值
- 明确的错误消息说明缺少哪个字段
- 正确处理步骤编号转换（1-based 到 0-based）

## 测试
创建了 `test_advanced_debugger.py` 测试脚本，可以：
1. 加载示例轨迹文件
2. 构建正确的输入数据
3. 调用 Advanced Debugger API
4. 显示分析结果

## 使用方法
```bash
# 设置 API Key
export OPENAI_API_KEY="your-api-key"

# 运行测试
python scripts/test_advanced_debugger.py

# 使用 Advanced Debugger
python scripts/rollout/openmanus_rollout_debugger.py \
    --env alfworld \
    --enable_debugger \
    --debugger_type advanced \
    --max_debug_retry 3
```

## 注意事项
1. 必须设置 `OPENAI_API_KEY` 环境变量
2. Advanced Debugger 需要完整的 chat_history，不能为空
3. 所有 API 调用失败都会抛出异常，不会返回默认值
4. 日志级别设置为 INFO 或 DEBUG 可以看到更多调试信息

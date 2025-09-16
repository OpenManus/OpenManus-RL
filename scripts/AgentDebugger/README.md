# Agent Error Detector Interface

## 更新说明 (2025-01-14)

本目录包含了Agent错误检测系统的最新版本，已更新以下内容：
- 使用最新的错误定义（error_definitions_loader_v3.py）
- 移除了错误严重性评分系统
- Phase 2采用全局视角分析
- 支持System和Others错误类别

## 文件说明

### 核心文件
- **api_interface_v2.py** - 最新API接口（推荐使用）
- **analysis_v5_error_detection.py** - Phase 1错误检测（最新版）
- **analysis_phase2_v3_critical.py** - Phase 2关键错误识别（最新版）
- **error_definitions_loader_v3.py** - 错误定义加载器（最新版）
- **error_type_definition.md** - 错误类型定义文档

### 旧版本文件（保留兼容性）
- api_interface.py - 旧版API接口
- analysis_v4_error_detection.py - 旧版Phase 1
- analysis_phase2_v2_critical.py - 旧版Phase 2

## 快速开始

### 1. 安装依赖
```bash
pip install aiohttp
```

### 2. 设置API密钥
```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 使用示例

```python
from api_interface_v2 import analyze_trajectory_sync
import json

# 加载轨迹
with open('trajectory.json', 'r') as f:
    trajectory = json.load(f)

# 分析轨迹
results = analyze_trajectory_sync(trajectory, "your-api-key")

# 获取关键错误
if results['critical_error']:
    critical = results['critical_error']
    print(f"Critical error at step {critical['critical_step']}")
    print(f"Error type: {critical['error_type']}")
    print(f"Correction guidance: {critical['correction_guidance']}")
```

## 错误类型更新

### Memory模块
- over_simplification
- memory_retrieval_failure
- hallucination

### Reflection模块
- progress_misjudge
- outcome_misinterpretation
- causal_misattribution
- hallucination

### Planning模块
- constraint_ignorance
- impossible_action
- inefficient_plan

### Action模块
- misalignment
- invalid_action
- format_error
- parameter_error

### System模块
- step_limit
- tool_execution_error
- llm_limit
- environment_error

### Others模块
- others（其他未定义错误）

## API函数

### 主要函数

#### analyze_trajectory_sync(trajectory_json, api_key)
完整分析（Phase 1 + Phase 2）

#### detect_errors_sync(trajectory_json, api_key)
仅Phase 1错误检测

#### find_critical_error_sync(phase1_results, trajectory_json, api_key)
仅Phase 2关键错误识别

#### get_error_definitions_sync(module=None)
获取错误定义（不需要API密钥）

## Phase 2关键改进

1. **无预设权重**：移除了0-1的严重性评分系统
2. **全局视角**：从整体考虑失败的根本原因
3. **System错误考虑**：step_limit等系统错误也可能是关键错误
4. **改进的JSON解析**：更好地处理格式错误的响应

## 输出格式

### Phase 1输出
```json
{
    "task_id": "unknown",
    "task_description": "put two soapbar in toilet",
    "task_success": false,
    "environment": "alfworld",
    "total_steps": 30,
    "step_analyses": [
        {
            "step": 1,
            "errors": {
                "memory": {...},
                "reflection": {...},
                "planning": {...},
                "action": {...}
            },
            "summary": "..."
        }
    ]
}
```

### Phase 2输出
```json
{
    "critical_error": {
        "critical_step": 5,
        "critical_module": "planning",
        "error_type": "impossible_action",
        "root_cause": "...",
        "evidence": "...",
        "correction_guidance": "...",
        "cascading_effects": [...],
        "confidence": 0.95
    }
}
```

## 注意事项

1. 确保使用`api_interface_v2.py`而不是旧版本
2. 模型默认使用`gpt-4.1-2025-04-14`
3. Phase 2现在从全局角度判断关键错误，不依赖预设权重
4. System错误（如step_limit）现在会被正确识别为潜在关键错误

## 联系方式

作者：Zijia Liu
仓库：https://github.com/m-serious/Agent-Error-Detector
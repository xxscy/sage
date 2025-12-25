# RAG COT 链模块化结构

## 模块说明

原 `rag_cot_chain.py` 文件已拆分为以下模块：

### 1. `config.py`
- `RAGCOTConfig`: 配置类

### 2. `tools.py`
- `ContextUnderstandingTool`: 上下文理解工具
- `DeviceLookupTool`: 设备查找工具
- `WeatherLookupTool`: 天气查找工具

### 3. `utils.py`
- 辅助函数（设备状态、VLM、设备类别等）

### 4. `testcase_loader.py`
- `TestCaseInfo`: 测试用例信息类
- `load_test_cases()`: 加载测试用例
- `extract_user_command_from_test()`: 提取用户命令

### 5. `prompts.py`
- `build_intent_analysis_prompt()`: 构建意图分析prompt
- `build_environment_overview_prompt()`: 构建环境概览prompt
- `build_cot_prompt()`: 构建COT prompt
- `build_chain_planner_prompt()`: 构建链式规划prompt

### 6. `parser.py`
- `parse_planner_response()`: 解析规划器响应
- `parse_llm_response()`: 解析LLM响应

### 7. `evaluator.py`
- `evaluate_test_case()`: 评估单个测试用例
- `run_rag_cot_evaluation()`: 运行完整评估
- `print_evaluation_summary()`: 打印评估摘要

### 8. `main.py`
- 主入口函数

## 使用方式

### 方式1：使用新模块（推荐）
```python
from sage.testing.rag_cot import run_rag_cot_evaluation, print_evaluation_summary
from sage.testing.rag_cot.config import RAGCOTConfig
from sage.utils.llm_utils import GPTConfig

config = RAGCOTConfig(llm_config=GPTConfig(model_name="gpt-4o-mini", temperature=0.0))
summary = run_rag_cot_evaluation(config)
print_evaluation_summary(summary)
```

### 方式2：使用主入口
```python
from sage.testing.rag_cot.main import main
main()
```

### 方式3：向后兼容（原文件仍可用）
```python
from sage.testing.rag_cot_chain import run_rag_cot_evaluation, print_evaluation_summary
# 原代码继续工作
```

## 拆分状态

⚠️ **注意**: 当前拆分工作正在进行中。部分模块可能仍在原文件中。

完成拆分后，所有功能将通过 `sage.testing.rag_cot` 模块提供。


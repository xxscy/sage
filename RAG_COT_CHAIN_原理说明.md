# RAG COT Chain 运行原理详解

## 📋 项目概述

`rag_cot_chain.py` 是一个**评估系统**，用于判断智能家居助手在什么情况下需要调用 `human_interaction_tool` 来向用户澄清问题。它结合了：

- **RAG (Retrieval-Augmented Generation)**: 检索增强生成，从数据库中检索用户偏好、历史交互和设备信息
- **COT (Chain of Thought)**: 思维链推理，让 LLM 进行多步骤推理判断

## 🎯 核心目标

**判断用户指令是否需要人工交互澄清**

例如：
- ✅ **不需要澄清**: "turn on the TV" - 指令明确
- ❌ **需要澄清**: "turn it off" - "it" 指代不明
- ❌ **需要澄清**: "put on my favorite show" - 缺少偏好信息

## 🏗️ 系统架构

### 1. 配置类：`RAGCOTConfig`

```python
@dataclass
class RAGCOTConfig:
    llm_config: LLMConfig          # LLM 配置（模型、温度等）
    test_types_to_include: List[str]  # 要评估的测试类型
    user_name: str                 # 默认用户名
    preference_query_template: str  # 偏好检索查询模板
    max_test_cases: Optional[int]   # 最大测试用例数
    device_lookup_max_results: int  # 设备检索最大结果数
```

### 2. 核心工具类

#### `ContextUnderstandingTool` - 上下文理解工具
- **作用**: 汇总所有检索到的上下文信息
- **输入**: 用户偏好、设备状态、历史片段、设备查找结果
- **输出**: 结构化的上下文摘要

#### `DeviceLookupTool` - 设备查找工具
- **作用**: 根据用户指令查找相关设备
- **原理**: 
  - 使用 `DocManager` 获取设备元数据（名称、能力、组件）
  - 结合 `fake_requests` 的设备状态快照
  - 通过关键词匹配和相似度计算找到最相关的设备
- **输出**: 匹配的设备列表（包含设备ID、能力、状态等）

### 3. 测试用例类：`TestCaseInfo`

```python
class TestCaseInfo:
    name: str                      # 测试用例名称
    user_command: str             # 用户指令
    types: List[str]              # 测试类型标签
    source_code: str              # 源代码
    requires_human_interaction: bool  # 是否真的需要人工交互（ground truth）
    device_state: Dict            # 设备状态快照
```

## 🔄 完整运行流程

### 阶段 1: 初始化与数据加载

```python
run_rag_cot_evaluation(config)
    ↓
1. load_test_cases()  # 从 testcases.py 加载所有测试用例
2. 过滤测试用例（排除 google/test_set 类型）
3. 初始化工具：
   - LLM 实例
   - UserProfileTool（用户偏好检索）
   - ContextUnderstandingTool
   - DeviceLookupTool
4. 准备 few-shot 示例
```

### 阶段 2: 对每个测试用例进行评估

```python
evaluate_test_case(test_case, ...)
```

#### 2.1 多步骤工具调用循环（最多 25 步）

这是一个**自主规划循环**，LLM 决定每一步要做什么：

```
┌─────────────────────────────────────────┐
│  步骤 1: 构建规划提示词                │
│  build_chain_planner_prompt()          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  步骤 2: LLM 决定下一步行动            │
│  可选行动：                              │
│  - preference_lookup: 检索用户偏好      │
│  - device_lookup: 查找相关设备          │
│  - context_summary: 生成上下文摘要      │
│  - final_decision: 进行最终判断         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  步骤 3: 执行选定的行动                 │
│  更新上下文状态                         │
└─────────────────────────────────────────┘
              ↓
         [循环直到 final_decision]
```

#### 2.2 规划提示词结构

`build_chain_planner_prompt()` 生成的提示词包含：

```
- 当前用户指令
- 已检索的上下文状态（偏好、设备、摘要）
- 可用工具列表
- 之前的步骤历史
- 要求：返回 JSON 格式的决策
```

LLM 返回格式：
```json
{
  "thought": "需要查找用户偏好信息",
  "action": "preference_lookup",
  "query": "What is the user's favorite TV show?"
}
```

#### 2.3 各行动的执行逻辑

**A. preference_lookup（检索用户偏好）**
```python
if action == "preference_lookup":
    query = planner_decision["query"] or preference_query
    tool_input = json.dumps({
        "query": query,
        "user_name": effective_user_name
    })
    user_preferences = user_profile_tool.run(tool_input)
    # 更新 user_preferences 变量
```

**B. device_lookup（查找设备）**
```python
if action == "device_lookup":
    query = planner_decision["query"] or test_case.user_command
    lookup_result = device_lookup_tool.run(query)
    device_facts.append(lookup_result)
    # 添加到设备事实列表
```

**C. context_summary（生成上下文摘要）**
```python
if action == "context_summary":
    context_summary = context_tool.run(
        user_command=test_case.user_command,
        user_preferences=user_preferences,
        device_state=test_case.device_state,
        user_memory_snippets=user_memory_snippets,
        device_lookup_notes=device_facts,
    )
```

**D. final_decision（最终判断）**
```python
if action == "final_decision":
    # 如果还没有上下文摘要，先生成一个
    if context_summary is None:
        context_summary = context_tool.run(...)
    
    # 构建 COT 推理提示词
    final_prompt = build_cot_prompt(
        test_case=test_case,
        user_preferences=user_preferences,
        user_memory_snippets=user_memory_snippets,
        context_summary=context_summary,
        device_lookup_notes=device_facts,
    )
    
    # LLM 进行最终推理
    final_response = llm([HumanMessage(content=final_prompt)])
    
    # 解析结果
    predicted_needs_tool, final_reasoning = parse_llm_response(
        final_response_text
    )
    break  # 退出循环
```

### 阶段 3: COT 推理提示词

`build_cot_prompt()` 构建的提示词包含：

1. **任务说明**: 判断是否需要 `human_interaction_tool`
2. **使用场景**:
   - 模糊指代（"it", "this", "that"）
   - 缺少个性化信息
   - 非标准表达（俚语、主观形容词）
   - 意图模糊
3. **不使用场景**:
   - 指令明确
   - 可以从数据库/历史中获取信息
   - 上下文已能消歧
4. **置信度启发式规则**:
   - 设备查找返回单一高匹配 → 使用该设备
   - 用户偏好提到类型/情绪 → 映射主观形容词
   - 环境抱怨 → 假设明显的执行器
5. **已检索的上下文**:
   - 用户偏好
   - 历史交互片段
   - 设备查找结果
   - 上下文摘要
6. **推理步骤要求**:
   ```
   Step 1 - Keyword analysis: 关键词分析
   Step 2 - Ambiguity check: 模糊性检查
   Step 3 - Information availability: 信息可用性
   Step 4 - Decision: 最终决策
   ```
7. **输出格式**:
   ```
   Conclusion:
   Need / Do not need to use human_interaction_tool
   ```

### 阶段 4: 结果解析与评估

```python
def parse_llm_response(response: str) -> Tuple[bool, str]:
    """
    解析 LLM 响应，提取是否需要 human_interaction_tool
    支持中英文格式
    """
    # 查找 "Conclusion: Need/Do not need" 格式
    # 或关键词匹配
    # 返回 (是否需要工具, 推理文本)
```

**评估逻辑**:
```python
is_correct = predicted_needs_tool == test_case.requires_human_interaction
```

### 阶段 5: 日志记录

每个测试用例的结果会保存为 JSON 文件：
```json
{
  "test_name": "turn_on_tv",
  "user_command": "Amal: turn on the TV",
  "ground_truth_requires_human_interaction": false,
  "predicted_requires_human_interaction": false,
  "is_correct": true,
  "reasoning": "...",
  "chain_history": [...],
  "device_lookup_notes": [...]
}
```

### 阶段 6: 统计汇总

```python
summary = {
    "total_cases": 42,
    "correct": 18,
    "accuracy": 0.4286,
    "help_accuracy": 0.75,      # 需要交互时的正确率
    "non_help_accuracy": 0.30,  # 不需要交互时的正确率
    "type_statistics": {...}     # 按类型统计
}
```

## 🔑 关键技术细节

### 1. 用户名提取

```python
def extract_user_name_from_command(command: str) -> Optional[str]:
    """
    从 "Amal: turn on the TV" 中提取 "amal"
    支持中英文冒号
    """
    match = re.match(r"\s*([A-Za-z]+)\s*[:：]", command)
    return match.group(1).strip().lower() if match else None
```

### 2. 设备查找算法

`DeviceLookupTool._search_devices()` 使用：

1. **关键词匹配**: 计算查询词在设备元数据中的出现次数
2. **相似度匹配**: 使用 `SequenceMatcher` 计算字符串相似度
3. **综合得分**: `score = token_score * 2 + ratio_score`
4. **过滤**: 如果 token_score=0 且 ratio_score<0.2，则跳过

### 3. 测试用例命令提取

`extract_user_command_from_test()` 从测试函数源代码中提取命令：

1. 优先查找 `coordinator.execute("...")` 中的直接字符串
2. 其次查找 `user_command = "..."` 或 `command = "..."` 变量
3. 支持单行、多行、三引号字符串

### 4. Few-shot 学习

系统会从测试用例中选择示例：
- 2 个需要人工交互的示例
- 2 个不需要人工交互的示例

这些示例会添加到 COT 提示词中，帮助 LLM 学习判断模式。

## 📊 数据流图

```
用户指令
    ↓
[提取用户名] → UserProfileTool → 用户偏好
    ↓
[设备查找] → DeviceLookupTool → 设备信息
    ↓
[上下文汇总] → ContextUnderstandingTool → 上下文摘要
    ↓
[COT 推理] → LLM → 最终判断
    ↓
[结果评估] → 正确/错误
```

## 🎓 设计理念

1. **自主规划**: LLM 自主决定需要检索哪些信息，而不是固定流程
2. **渐进式信息收集**: 通过多步骤循环逐步收集必要信息
3. **避免过度询问**: 系统偏向于自主解决，只在真正需要时才询问用户
4. **上下文感知**: 充分利用用户偏好、历史交互和设备状态来消歧

## 🚀 使用示例

```python
from sage.testing.rag_cot_chain import RAGCOTConfig, run_rag_cot_evaluation, print_evaluation_summary
from sage.utils.llm_utils import GPTConfig

# 配置
config = RAGCOTConfig(
    llm_config=GPTConfig(
        model_name="gpt-4o-mini",
        temperature=0.0
    ),
    max_test_cases=10,  # 只评估前 10 个测试用例
    test_types_to_include=["ambiguous"]  # 只评估模糊类型
)

# 运行评估
summary = run_rag_cot_evaluation(config)

# 打印结果
print_evaluation_summary(summary)
```

## 📝 总结

这个系统通过**RAG + COT**的方式，让 LLM 能够：
1. 自主决定需要检索哪些信息
2. 逐步收集用户偏好、设备信息、历史上下文
3. 基于完整上下文进行推理判断
4. 准确识别何时需要向用户澄清问题

这是一个典型的**工具使用（Tool Use）**和**自主规划（Autonomous Planning）**的应用场景。




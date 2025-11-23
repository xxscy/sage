"""
RAG 系统的 COT 链，用于判断是否需要使用 human_interaction_tool 工具。

该模块从 testcases.py 中读取测试用例，先根据用户指令构造检索请求，
检索与该用户相关的偏好信息和设备信息，然后在包含这些上下文的
中文 COT prompt 中，让 LLM 判断是否需要使用 human_interaction_tool
来澄清用户问题。
"""
import inspect
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from langchain.schema.messages import HumanMessage

from sage.testing.testcases import TEST_CASE_TYPES, get_tests
from sage.utils.llm_utils import GPTConfig, LLMConfig
from sage.utils.common import CONSOLE
from sage.retrieval.tools import UserProfileToolConfig
from sage.testing.testing_utils import get_base_device_state
from sage.smartthings.docmanager import DocManager


@dataclass
class RAGCOTConfig:
    """RAG COT链的配置"""
    llm_config: LLMConfig = GPTConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        streaming=False,
    )
    test_types_to_include: List[str] = None  # None表示包含所有类型
    # 如果不指定，将尝试从用户指令前缀中自动解析（如 "Abhisek : xxx"）
    user_name: str = "default_user"
    # 每条指令用于检索用户偏好的查询模板（发送给 LLM，使用英文）
    preference_query_template: str = (
        "Based on this command, infer what preference-related query should be used "
        "for retrieval: {user_command}"
    )
    max_test_cases: Optional[int] = None  # 限制本次评估的测试用例数量
    device_lookup_max_results: int = 3


class ContextUnderstandingTool:
    """总结用户偏好、历史与设备环境的轻量工具"""

    def _summarize_mapping(self, data: Any, limit: int = 5) -> str:
        if not isinstance(data, dict):
            return str(data) if data else "(empty)"

        pairs = []
        for idx, (key, value) in enumerate(data.items()):
            if idx >= limit:
                pairs.append("...")
                break
            pairs.append(f"{key}: {value}")
        return "; ".join(pairs) if pairs else "(empty)"

    def run(
        self,
        *,
        user_command: str,
        user_preferences: Any,
        device_state: Optional[Dict[str, Any]],
        user_memory_snippets: Optional[List[str]],
        device_lookup_notes: Optional[List[str]] = None,
    ) -> str:
        preference_summary = self._summarize_mapping(user_preferences)
        device_summary = self._summarize_mapping(device_state or {})
        history_summary = (
            " | ".join(user_memory_snippets)
            if user_memory_snippets
            else "(no history snippets)"
        )
        device_lookup_summary = (
            " | ".join(device_lookup_notes)
            if device_lookup_notes
            else "(device lookup not used)"
        )

        summary_sections = [
            f"Command focus: {user_command}",
            f"Preference snapshot: {preference_summary}",
            f"Device environment: {device_summary}",
            f"History context: {history_summary}",
            f"Device lookup insights: {device_lookup_summary}",
        ]

        return "\n".join(summary_sections)


class DeviceLookupTool:
    """基于 DocManager 与 fake_requests 设备快照的轻量检索工具"""

    def __init__(self, *, docmanager_cache_path: Path, max_results: int = 3):
        self.doc_manager = DocManager.from_json(docmanager_cache_path)
        self.max_results = max_results
        self.device_state: Dict[str, Any] = {}

    def update_device_state(self, device_state: Optional[Dict[str, Any]]):
        self.device_state = device_state or {}

    def run(self, query: str) -> str:
        if not query:
            return "DeviceLookupTool: query is empty."
        return self._search_devices(query)

    def _search_devices(self, query: str) -> str:
        available_ids = list(self.device_state.keys()) or list(
            self.doc_manager.default_devices
        )
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if t]
        matches: List[Tuple[float, str, Dict[str, Any]]] = []

        for device_id in available_ids:
            metadata = self._build_device_metadata(device_id)
            blob = metadata["search_blob"]
            if not blob:
                continue

            token_score = sum(1 for token in tokens if token in blob)
            ratio_score = SequenceMatcher(None, query.lower(), blob[:5000]).ratio()
            # 轻量得分：token匹配更高权重
            score = token_score * 2 + ratio_score
            if token_score == 0 and ratio_score < 0.2:
                continue
            matches.append((score, device_id, metadata))

        if not matches:
            return f"DeviceLookupTool: 未找到与 \"{query}\" 相关的设备。"

        matches.sort(key=lambda item: item[0], reverse=True)
        limited = matches[: self.max_results]
        lines = []
        for score, device_id, metadata in limited:
            lines.append(
                self._format_device_line(
                    device_id=device_id,
                    metadata=metadata,
                    score=score,
                )
            )

        extra = len(matches) - len(limited)
        if extra > 0:
            lines.append(f"(另有 {extra} 个候选被截断)")
        return "\n".join(lines)

    def _build_device_metadata(self, device_id: str) -> Dict[str, Any]:
        device_name = self.doc_manager.device_names.get(device_id, "Unknown device")
        capabilities = [
            cap["capability_id"]
            for cap in self.doc_manager.device_capabilities.get(device_id, [])
        ]
        components = list(self.device_state.get(device_id, {}).keys())
        search_blob = " ".join(
            [
                device_id.lower(),
                device_name.lower(),
                " ".join(capabilities).lower(),
                " ".join(components).lower(),
            ]
        )
        return {
            "name": device_name,
            "capabilities": capabilities,
            "components": components,
            "state_summary": self._summarize_device_state(device_id),
            "search_blob": search_blob,
        }

    def _summarize_device_state(self, device_id: str, component_limit: int = 2) -> str:
        device_state = self.device_state.get(device_id)
        if not device_state:
            return "(no fake_requests snapshot for this device)"

        component_chunks = []
        for comp_idx, (component, cap_dict) in enumerate(device_state.items()):
            if comp_idx >= component_limit:
                component_chunks.append("...")
                break
            cap_chunks = []
            for cap_idx, (capability, attributes) in enumerate(cap_dict.items()):
                if cap_idx >= 3:
                    cap_chunks.append("...")
                    break
                attr_summary = self._format_attribute_values(attributes)
                cap_chunks.append(f"{capability}: {attr_summary}")
            component_chunks.append(f"{component} -> {' | '.join(cap_chunks)}")
        return " || ".join(component_chunks) if component_chunks else "(empty component list)"

    def _format_attribute_values(self, attributes: Any, limit: int = 2) -> str:
        if isinstance(attributes, dict):
            attr_items = list(attributes.items())
            formatted = []
            for idx, (attr_name, attr_value) in enumerate(attr_items):
                if idx >= limit:
                    formatted.append("...")
                    break
                formatted.append(f"{attr_name}={self._extract_value(attr_value)}")
            return ", ".join(formatted) if formatted else "(empty capability)"
        return str(attributes)

    def _extract_value(self, attr_value: Any) -> Any:
        if isinstance(attr_value, dict):
            if "value" in attr_value:
                return attr_value["value"]
            # 返回最外层键，避免超长
            keys = list(attr_value.keys())
            if not keys:
                return "{}"
            preview_key = keys[0]
            return {preview_key: attr_value[preview_key]}
        return attr_value

    def _format_device_line(
        self,
        *,
        device_id: str,
        metadata: Dict[str, Any],
        score: float,
    ) -> str:
        capability_preview = ", ".join(metadata["capabilities"][:5]) or "(no capability data)"
        if len(metadata["capabilities"]) > 5:
            capability_preview += ", ..."
        component_preview = ", ".join(metadata["components"][:3]) or "(components unavailable)"
        state_summary = metadata["state_summary"]
        return (
            f"[匹配度 {score:.2f}] {metadata['name']} ({device_id}) | "
            f"Capabilities: {capability_preview} | Components: {component_preview} | "
            f"State: {state_summary}"
        )


def _sanitize_filename(name: str) -> str:
    """将任意字符串转换为安全的文件/文件夹名"""
    # 替换不安全字符
    safe = re.sub(r"[^\w\-_. ]", "_", name)
    # 去掉首尾空格并限制长度
    safe = safe.strip()[:80]
    return safe or "empty_command"


def _summarize_device_state(device_state: Optional[Dict[str, Any]]) -> str:
    """生成简短的设备状态摘要，避免输出整个大字典。"""
    if not device_state:
        return "(no device state)"
    if isinstance(device_state, dict):
        keys = list(device_state.keys())
        sample_keys = ", ".join(keys[:5])
        if len(keys) > 5:
            sample_keys += ", ..."
        return f"{len(keys)} entries: {sample_keys}"
    return f"(device state type: {type(device_state).__name__})"


def extract_user_name_from_command(command: str) -> Optional[str]:
    """
    从用户指令中提取用户名。

    约定格式类似于：
        "Amal: turn on the TV"
        "Abhisek : what did I miss?"
        "dmitriy: put the game on the tv"
    """
    if not command:
        return None

    # 匹配前缀名字，后面跟一个冒号（支持中英文冒号），中间允许有空格
    match = re.match(r"\s*([A-Za-z]+)\s*[:：]", command)
    if not match:
        return None

    return match.group(1).strip().lower()


def _extract_user_name_from_command(command: str) -> Optional[str]:
    """
    从用户指令中解析用户名前缀，例如：
    "Abhisek : what did I miss?" -> "abhisek"
    "Amal: turn on the TV" -> "amal"
    """
    if not command:
        return None
    # 只看第一个冒号前面的内容
    parts = command.split(":", 1)
    if len(parts) < 2:
        return None
    candidate = parts[0].strip()
    if not candidate:
        return None
    # 只保留首个“单词”，并转小写
    return candidate.split()[0].lower()


class TestCaseInfo:
    """测试用例信息"""
    def __init__(
        self,
        name: str,
        user_command: str,
        types: List[str],
        source_code: str,
        device_state: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.user_command = user_command
        self.types = types
        self.source_code = source_code
        self.requires_human_interaction = "human_interaction" in types
        # 离线评估时的设备状态快照（来自 testing_utils 中的 device_state pickle）
        self.device_state = device_state or {}


def extract_user_command_from_test(test_func) -> str:
    """
    从测试函数中提取用户命令
    
    查找 coordinator.execute() 调用中的字符串参数
    优先查找直接传入execute的字符串，其次查找变量赋值
    对于使用字符串拼接的execute调用，查找前面的command/user_command变量
    """
    source = inspect.getsource(test_func)
    
    # 首先尝试查找 coordinator.execute("...") 或 coordinator.execute('...')
    # 支持单行和多行字符串
    # 匹配模式：coordinator.execute("...") 或 coordinator.execute(\n    "..."\n)
    execute_patterns = [
        # 单行：coordinator.execute("...")
        r'coordinator\.execute\(\s*["\']([^"\']+)["\']\s*\)',
        # 多行：coordinator.execute(\n    "..."\n) - 更灵活的匹配
        r'coordinator\.execute\(\s*\n\s*["\']([^"\']+)["\']\s*\n?\s*\)',
        # 三引号字符串（多行）
        r'coordinator\.execute\(\s*"""([^"]+)"""\s*\)',
        r"coordinator\.execute\(\s*'''([^']+)'''\s*\)",
        # 多行三引号
        r'coordinator\.execute\(\s*\n\s*"""([^"]+)"""\s*\n?\s*\)',
        r"coordinator\.execute\(\s*\n\s*'''([^']+)'''\s*\n?\s*\)",
    ]
    
    # 找到所有execute调用，按出现顺序排序
    all_execute_matches = []
    for pattern in execute_patterns:
        matches = re.finditer(pattern, source, re.MULTILINE | re.DOTALL)
        for match in matches:
            command = match.group(1).strip()
            if command:  # 确保不是空字符串
                all_execute_matches.append((match.start(), command))
    
    if all_execute_matches:
        # 按位置排序，取第一个（最早的）execute调用中的命令
        all_execute_matches.sort(key=lambda x: x[0])
        return all_execute_matches[0][1]
    
    # 如果没找到直接字符串，查找变量赋值
    # 查找 user_command = "..." 或 command = "..."
    # 优先查找在第一个execute调用之前的变量赋值
    variable_patterns = [
        r'user_command\s*=\s*["\']([^"\']+)["\']',
        r'command\s*=\s*["\']([^"\']+)["\']',
        # 支持多行字符串赋值
        r'user_command\s*=\s*"""([^"]+)"""',
        r'command\s*=\s*"""([^"]+)"""',
        r"user_command\s*=\s*'''([^']+)'''",
        r"command\s*=\s*'''([^']+)'''",
        # 支持括号包裹的多行字符串
        r'user_command\s*=\s*\(\s*["\']([^"\']+)["\']\s*\)',
        r'command\s*=\s*\(\s*["\']([^"\']+)["\']\s*\)',
    ]
    
    # 找到第一个execute调用的位置
    first_execute_match = re.search(r'coordinator\.execute\(', source, re.DOTALL)
    first_execute_pos = first_execute_match.start() if first_execute_match else len(source)
    
    # 查找在第一个execute调用之前的变量赋值
    best_match = None
    best_pos = -1
    
    for pattern in variable_patterns:
        matches = re.finditer(pattern, source, re.DOTALL)
        for match in matches:
            if match.start() < first_execute_pos:  # 在第一个execute调用之前
                command = match.group(1).strip()
                if command and match.start() > best_pos:
                    best_match = command
                    best_pos = match.start()
    
    if best_match:
        return best_match
    
    # 如果还是没找到，尝试查找 coordinator.execute() 调用，可能使用变量
    # 这种情况下，我们尝试找到最近的字符串赋值
    execute_match = re.search(r'coordinator\.execute\(([^)]+)\)', source, re.DOTALL)
    if execute_match:
        var_name = execute_match.group(1).strip()
        # 尝试找到这个变量的赋值
        var_pattern = rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']'
        var_match = re.search(var_pattern, source, re.DOTALL)
        if var_match:
            return var_match.group(1).strip()
    
    return None


def load_test_cases() -> List[TestCaseInfo]:
    """
    从testcases.py加载所有测试用例信息
    """
    from sage.testing import testcases

    test_cases: List[TestCaseInfo] = []
    # 获取所有注册的测试函数
    all_tests = get_tests(list(testcases.TEST_REGISTER.keys()), combination="union")

    # 离线评估时统一使用基础设备状态快照
    base_device_state = get_base_device_state()

    skipped_cases = []  # 记录被跳过的测试用例
    
    for test_func in all_tests:
        test_name = test_func.__name__
        types = list(TEST_CASE_TYPES.get(test_name, []))
        user_command = extract_user_command_from_test(test_func)
        source_code = inspect.getsource(test_func)

        if user_command:
            test_cases.append(
                TestCaseInfo(
                    name=test_name,
                    user_command=user_command,
                    types=types,
                    source_code=source_code,
                    device_state=base_device_state,
                )
            )
        else:
            skipped_cases.append((test_name, types))
    
    if skipped_cases:
        CONSOLE.log(f"[yellow]警告: {len(skipped_cases)} 个测试用例无法提取用户命令，已跳过:")
        for name, types in skipped_cases:
            CONSOLE.log(f"  - {name} (types: {types})")

    return test_cases


def build_cot_prompt(
    test_case: TestCaseInfo,
    user_preferences: Optional[Dict[str, Any]] = None,
    user_memory_snippets: Optional[List[str]] = None,
    examples: List[TestCaseInfo] = None,  # 保留参数以保持兼容性，但不再使用（零样本模式）
    context_summary: Optional[str] = None,
    device_lookup_notes: Optional[List[str]] = None,
) -> str:
    """
    构建 COT（Chain of Thought）推理提示词。

    LLM 的工作流程应当是：
    1. 先根据给定的用户指令，理解需要检索哪些用户偏好和设备信息；
    2. 然后在已经提供好的「用户偏好、历史交互片段、当前设备状态」上下文基础上，
       判断是否需要调用 human_interaction_tool 进一步向用户澄清。

    Args:
        test_case: 当前要判断的测试用例
        user_preferences: 针对当前 user_name 检索到的用户偏好摘要
        user_memory_snippets: 针对当前 user_name 和命令检索到的历史交互片段
        examples: 示例测试用例（已弃用，使用零样本模式）

    Returns:
        构建好的 prompt 字符串（英文，用于发送给 LLM）
    """
    preferences_text = (
        "(no user preference information available)"
        if not user_preferences
        else str(user_preferences)
    )
    memory_text = (
        "(no historical interactions related to the current command)"
        if not user_memory_snippets
        else "\n".join(user_memory_snippets)
    )
    device_lookup_text = (
        "(device lookup tool not used)"
        if not device_lookup_notes
        else "\n".join(device_lookup_notes)
    )
    context_summary_text = (
        context_summary if context_summary else "(context summary unavailable)"
    )

    base_prompt = f"""You are the decision module of a smart home assistant.
Only the information retrieved by earlier planner/tool calls is available to you now.
Do not assume that the full list of devices was preloaded; rely solely on the targeted
facts that have already been fetched.

Your task:
1. Using the already retrieved user preferences, history snippets, and device state, understand the current user command.
2. Apply deep reasoning to infer the user's underlying intent and situational context.
3. Decide whether you need to call the `human_interaction_tool` to ask the user clarifying questions.

CRITICAL: Deep Intent Inference
Before deciding to ask for clarification, you MUST perform deep reasoning about the user's situation and intent:

1. **Situational Context Analysis**: Consider the broader context implied by the command.
   - Identify situational cues: incoming communications (calls, messages), time-based activities (sleep, meals, work), environmental states (too bright/cold/hot/loud), or activity transitions.
   - Infer corrective intent: When users mention problems or transitions, they typically want actions that address the issue or facilitate the transition.
   - Apply reverse logic: Complaints about environmental conditions imply the user wants the opposite state (too bright → dimmer, too loud → quieter, too cold → warmer).

2. **Common Sense Reasoning**: Apply everyday logic and social conventions.
   - Communication interference: During active communication, audio adjustments typically aim to reduce interference (lower volume, mute).
   - Environmental complaints: Negative qualifiers ("too X") indicate desire for opposite adjustment.
   - Activity-based preferences: Activities like sleeping, working, or relaxing have associated environmental preferences (darker/quieter for sleep, appropriate lighting for work).
   - Task-appropriate settings: When users request "appropriate" or "suitable" settings for a task, infer settings that match the task's requirements (intensive cleaning for heavy soil, comfortable levels for general use).

3. **Default Action Inference**: When a specific value is missing but the intent is clear, infer a reasonable default action.
   - Directional adjustments: If direction is clear (reduce, increase, dim, brighten), apply moderate changes (typically 30-50% of current value or range).
   - Task-appropriate modes: For tasks requiring specific modes, select the mode that best matches the task characteristics (intensive for heavy-duty, gentle for delicate).
   - Comfort thresholds: For subjective terms like "comfortable" or "appropriate", use industry-standard defaults or mid-range values.

4. **Contextual Disambiguation**: Use the situation to resolve ambiguity.
   - Pronoun resolution: Resolve pronouns ("it", "this", "that") by matching to recently mentioned devices, active devices, or contextually prominent items.
   - Implicit references: When users refer to items without explicit naming ("the game", "that show"), match to recent context, active content, or user history.
   - Temporal continuity: Assume references maintain continuity with recent interactions unless explicitly contradicted.

Use `human_interaction_tool` ONLY in these situations:
1. Ambiguous reference: pronouns like "it", "this", "that" cannot be resolved even with deep reasoning and context.
2. Missing personalization info: critical user preferences are missing AND cannot be inferred from common sense (e.g., "my favourite color" when no color preference exists).
3. Non-standard expressions: truly unclear slang or creative wording that defies reasonable interpretation (uncommon neologisms, regional slang, or creative expressions without sufficient context to infer meaning).
4. Multiple valid interpretations: the command genuinely has multiple reasonable meanings that cannot be distinguished by context.

Do NOT use `human_interaction_tool` when:
1. The command is explicit and provides all necessary details.
2. Common sense and situational context allow you to infer the intent (directional adjustments during specific situations, environmental corrections, activity-based preferences).
3. You can apply reasonable defaults based on the situation (task-appropriate modes, comfort-level settings, moderate adjustments).
4. The context (device state, recent interactions, situational cues) disambiguates the command.
5. The missing information is a specific value, but the direction/intent is clear (adjustment direction is inferable, task requirements are clear).

Enhanced Confidence Heuristics:
- **Situational inference**: If the command mentions a situation (communication events, activity transitions, environmental states), infer the obvious corrective or facilitative action based on common sense.
- **Directional inference**: If the command specifies a direction (adjust, change, set) but not a value, infer the direction from context (reduce interference during communication, correct environmental extremes, facilitate activity transitions).
- **Default value inference**: When direction is clear but value is missing, use reasonable defaults (moderate percentage adjustments, task-appropriate intensity levels, industry-standard comfort ranges).
- **Device disambiguation**: If device lookup returns a single high-scoring match, treat it as the target.
- **Preference mapping**: Map subjective terms to user preferences when available, or to common defaults when not.
- **Environmental complaints**: Assume the obvious corrective action (reverse the complained condition: too bright→dimmer, too loud→quieter, too cold→warmer).
- **Status queries**: Answer directly from device state—no preferences needed.

Your STRONG bias should be to solve the command autonomously using deep reasoning and common sense. Only ask for clarification when there is a genuine, unresolvable ambiguity that prevents action.

Below is the information already retrieved for you (do NOT call additional tools; reason only with what is provided):

- Retrieved user preference / device facts (results of previous `preference_lookup` calls, may be empty):
{preferences_text}

- Retrieved historical interaction snippets (most relevant to the current command, may be empty):
{memory_text}

- Device lookup findings (from fake_requests snapshot + DocManager metadata):
{device_lookup_text}

- Context understanding summary (generated from targeted retrievals):
{context_summary_text}

Using the above context, analyze the user command with explicit Chain-of-Thought reasoning:

User command: "{test_case.user_command}"

Reasoning steps:
1. **Surface-level analysis**: Identify key verbs, nouns, adjectives, pronouns, and explicit parameters in the command.
2. **Deep intent inference**: Analyze the situational context and underlying user intent. What is the user really trying to achieve? What does the situation imply about the desired action?
   - Consider: What is happening in the user's environment? (communication events, activity transitions, environmental states, task contexts)
   - Infer: What action would make sense in this situation? (reducing interference, facilitating transitions, correcting environmental issues, matching task requirements)
   - Apply: Common sense reasoning about what the user likely wants based on situational patterns and social conventions.
3. **Information gap analysis**: Identify what specific information is missing (if any). Can this gap be filled by:
   - Common sense inference? (directional adjustments during specific situations, task-appropriate settings)
   - Situational context? (environmental corrections, activity-based preferences)
   - Reasonable defaults? (moderate adjustments, industry-standard values, task-appropriate modes)
   - Retrieved preferences/history? (user-specific preferences from database, historical patterns)
4. **Ambiguity resolution**: Determine if any remaining ambiguity is truly blocking or can be resolved through inference.
5. **Decision**: Decide whether to call `human_interaction_tool`. Default to "Do not need" when deep reasoning and common sense can resolve the command.

Format your answer exactly as:
```
Reasoning:
Step 1 - Surface-level analysis: [...]
Step 2 - Deep intent inference: [Analyze the situation and infer what the user really wants. Apply common sense reasoning about the context.]
Step 3 - Information gap analysis: [Identify missing information and determine if it can be inferred from context, common sense, or defaults.]
Step 4 - Ambiguity resolution: [Determine if any ambiguity is truly blocking.]
Step 5 - Decision: [...]

Conclusion:
Need / Do not need to use human_interaction_tool

Reason:
[Brief explanation in 1–2 sentences]
```

IMPORTANT: The line after `Conclusion:` MUST be exactly either `Need to use human_interaction_tool`
or `Do not need to use human_interaction_tool`. Any other wording is invalid.
"""
    
    # 使用零样本（zero-shot）prompt，不提供示例，确保评估公平性
    return base_prompt


def build_chain_planner_prompt(
    *,
    test_case: TestCaseInfo,
    chain_history: List[Dict[str, Any]],
    preference_query_template: str,
    context_state: Dict[str, Any],
) -> str:
    """
    Prompt the LLM to decide the next action in the multi-step chain.
    """
    history_text = (
        "(no previous steps taken)"
        if not chain_history
        else "\n".join(
            [
                f"Step {step['step']}: action={step['action']} | reasoning={step['reasoning']}"
                for step in chain_history
            ]
        )
    )

    preference_status = (
        "available" if context_state.get("user_preferences") else "missing"
    )
    summary_status = (
        "available" if context_state.get("context_summary") else "missing"
    )

    device_fact_status = (
        "available" if context_state.get("device_facts") else "missing"
    )

    planner_prompt = f"""You are orchestrating a Chain-of-Thought tool-using loop for the smart home assistant.
Nothing besides the user command is preloaded. When you need device-specific facts or user preferences,
issue focused retrieval instructions via the available tools instead of trying to enumerate the entire home.
Gather only what is necessary, then hand off to the final decision prompt that determines whether the
`human_interaction_tool` is required.

IMPORTANT: Deep Reasoning First
Before requesting additional information, apply deep reasoning about the user's situation and intent:
- Analyze the situational context: What is happening? (communication events, activity transitions, environmental states, task contexts)
- Infer the underlying intent: What does the user really want? (reducing interference, facilitating transitions, correcting environmental issues, matching task requirements)
- Apply common sense: Use everyday logic to fill gaps (directional adjustments during specific situations, environmental corrections, task-appropriate settings)
- Only retrieve information that is truly needed and cannot be inferred

Confidence & efficiency guidelines:
- **Reason before retrieving**: Apply deep reasoning and common sense before asking for more information.
- **Situational inference**: If the command mentions a situation (communication events, activity transitions, environmental states), infer the obvious action first.
- **Prefer a single decisive pass**: Once a tool supplies clear device or preference data, reuse it instead of repeating the same lookup.
- **Avoid redundant queries**: Don't ask for information that can be inferred from context or common sense.
- **Move to final_decision quickly**: As soon as the intent can be satisfied with existing evidence or inference, proceed to decision.
- **Autonomy over clarification**: The objective is to keep the assistant autonomous—treat clarification as a last resort.

Current user command: "{test_case.user_command}"

Retrieved context so far:
- User preference / device facts: {preference_status}
- Device lookup insights: {device_fact_status}
- Context summary: {summary_status}

Available tools (call at most one per step):
1. preference_lookup -> parameters: {{"query": "<english query string>"}}.
   Use for any missing user preference, historical insight, or device metadata.
   If you do not provide `query`, the system will reuse the default template:
   "{preference_query_template}".
2. device_lookup -> parameters: {{"query": "<english description of the target device/action>"}}.
   Surfaces relevant SmartThings device IDs, capabilities, and the latest fake_requests device state
   snapshot to help ground ambiguous references like "it" or "the fridge".
3. context_summary -> no parameters. Generates a consolidated summary using the latest
   user command, retrieved preferences/device facts, and history snippets.
4. final_decision -> no parameters. Use ONLY when you have enough information to
   produce the final answer. This will trigger a dedicated decision prompt, so do not
   include the final Need/Do not need wording here.

Previous steps:
{history_text}

Respond in strict JSON format with keys:
{{
  "thought": "<brief reasoning>",
  "action": "<preference_lookup | device_lookup | context_summary | final_decision>",
  "query": "<optional query string for preference_lookup>"
}}

Do NOT include any additional text outside the JSON."""
    return planner_prompt


def parse_planner_response(response_text: str) -> Dict[str, Any]:
    """Parse planner JSON; default to final_decision on failure."""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {
            "thought": response_text.strip(),
            "action": "final_decision",
            "query": None,
        }

    try:
        data = json.loads(json_match.group(0))
        action = data.get("action", "final_decision")
        if action not in {
            "preference_lookup",
            "device_lookup",
            "context_summary",
            "final_decision",
        }:
            action = "final_decision"
        return {
            "thought": data.get("thought", "").strip(),
            "action": action,
            "query": data.get("query"),
        }
    except json.JSONDecodeError:
        return {
            "thought": response_text.strip(),
            "action": "final_decision",
            "query": None,
        }


def parse_llm_response(response: str) -> Tuple[bool, str]:
    """
    解析LLM的响应，提取是否需要human_interaction_tool
    
    Returns:
        (是否需要human_interaction_tool, 推理过程)
    """
    response_lower = response.lower()
    
    # 查找中文或英文结论部分
    conclusion_match = re.search(
        r'(结论[：:]\s*(需要|不需要))|(Conclusion[：:]\s*(Need|Do\s*not\s*need))',
        response,
        re.IGNORECASE,
    )
    if conclusion_match:
        if conclusion_match.group(2):  # 中文匹配
            needs_tool = conclusion_match.group(2) == "需要"
        else:  # 英文匹配
            needs_tool = conclusion_match.group(4).lower().startswith("need")
        return needs_tool, response

    # 如果没有找到明确的结论格式，尝试关键词匹配（兼容中英文）
    if "human_interaction" in response_lower:
        if "need" in response_lower and "do not need" not in response_lower:
            return True, response
        if "do not need" in response_lower or "don't need" in response_lower:
            return False, response

    if "需要" in response and "human_interaction" in response_lower:
        if "不需要" not in response[:response.find("需要")]:
            return True, response
    if "不需要" in response or "不需要使用" in response:
        return False, response
    
    # 默认返回False
    return False, response


def evaluate_test_case(
    test_case: TestCaseInfo,
    llm,
    user_profile_tool,
    context_tool: ContextUnderstandingTool,
    device_lookup_tool: DeviceLookupTool,
    preference_query_template: str,
    user_name: Optional[str] = None,  # 保留参数以保持兼容性，但不再使用（优先从命令中提取）
    examples: List[TestCaseInfo] = None,  # 保留参数以保持兼容性，但不再使用（零样本模式）
    log_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    评估单个测试用例，判断 LLM 是否正确识别是否需要 human_interaction_tool。

    在单个测试用例内部运行多轮「工具-思考-再行动」循环，直到 LLM 触发最终决策。
    """
    preference_query = preference_query_template.format(
        user_command=test_case.user_command
    )

    # 优先从命令中提取用户名，不使用默认值
    parsed_user_name = extract_user_name_from_command(test_case.user_command)
    if parsed_user_name:
        effective_user_name = parsed_user_name.strip()  # extract_user_name_from_command 已返回小写
        CONSOLE.log(f"[cyan]从命令中提取到用户名: {effective_user_name}")
    else:
        effective_user_name = ""
        CONSOLE.log(f"[yellow]警告: 无法从命令中提取用户名: {test_case.user_command}")

    user_preferences: Any = {}
    user_memory_snippets: List[str] = []
    device_facts: List[str] = []
    context_summary: Optional[str] = None

    chain_history: List[Dict[str, Any]] = []
    planner_steps_limit = 25
    final_response_text = ""
    final_reasoning = ""
    predicted_needs_tool = False

    CONSOLE.rule(f"[bold blue]测试用例: {test_case.name}")
    CONSOLE.log(f"[bold]用户指令[/bold]: {test_case.user_command}")
    device_state_summary = _summarize_device_state(test_case.device_state)
    CONSOLE.log(f"[bold]设备状态摘要[/bold]: {device_state_summary}")

    for step_idx in range(1, planner_steps_limit + 1):
        planner_prompt = build_chain_planner_prompt(
            test_case=test_case,
            chain_history=chain_history,
            preference_query_template=preference_query,
            context_state={
                "user_preferences": user_preferences,
                "context_summary": context_summary,
                "device_facts": device_facts,
            },
        )
        planner_message = HumanMessage(content=planner_prompt)
        planner_response = llm([planner_message])
        planner_text = (
            planner_response.content
            if hasattr(planner_response, "content")
            else str(planner_response)
        )
        planner_decision = parse_planner_response(planner_text)
        action = planner_decision["action"]
        chain_history.append(
            {
                "step": step_idx,
                "action": action,
                "reasoning": planner_decision["thought"],
                "raw_response": planner_text,
            }
        )

        CONSOLE.log(
            f"[cyan]链式推理步骤 {step_idx}[/cyan]: action={action}, "
            f"thought={planner_decision['thought']}"
        )

        if action == "preference_lookup":
            query = planner_decision["query"] or preference_query
            CONSOLE.log(f"[yellow]调用 UserProfileTool，query: {query}")
            try:
                tool_input = json.dumps(
                    {"query": query, "user_name": effective_user_name},
                    ensure_ascii=False,
                )
                user_preferences = user_profile_tool.run(tool_input)
                CONSOLE.log(f"[green]User preferences 更新: {user_preferences}")
            except Exception as exc:
                user_preferences = "(failed to retrieve user preferences)"
                CONSOLE.log(
                    f"[red]UserProfileTool 调用失败: {exc}"
                )
            continue

        if action == "device_lookup":
            query = planner_decision["query"] or test_case.user_command
            lookup_result = device_lookup_tool.run(query)
            device_facts.append(lookup_result)
            CONSOLE.log(f"[green]设备检索结果[/green]: {lookup_result}")
            continue

        if action == "context_summary":
            context_summary = context_tool.run(
                user_command=test_case.user_command,
                user_preferences=user_preferences,
                device_state=test_case.device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
            CONSOLE.log(f"[green]上下文摘要更新: {context_summary}")
            continue

        # final_decision 或 fallback
        if context_summary is None:
            context_summary = context_tool.run(
                user_command=test_case.user_command,
                user_preferences=user_preferences,
                device_state=test_case.device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
            CONSOLE.log(f"[green]最终决策前生成上下文摘要: {context_summary}")

        final_prompt = build_cot_prompt(
            test_case=test_case,
            user_preferences=user_preferences,
            user_memory_snippets=user_memory_snippets,
            examples=None,  # 零样本模式，不提供示例
            context_summary=context_summary,
            device_lookup_notes=device_facts,
        )
        final_message = HumanMessage(content=final_prompt)
        final_response = llm([final_message])
        final_response_text = (
            final_response.content
            if hasattr(final_response, "content")
            else str(final_response)
        )
        predicted_needs_tool, final_reasoning = parse_llm_response(
            final_response_text
        )
        break
    else:
        # 达到最大步数仍未做最终判断，强制执行一次
        if context_summary is None:
            context_summary = context_tool.run(
                user_command=test_case.user_command,
                user_preferences=user_preferences,
                device_state=test_case.device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
        final_prompt = build_cot_prompt(
            test_case=test_case,
            user_preferences=user_preferences,
            user_memory_snippets=user_memory_snippets,
            examples=None,  # 零样本模式，不提供示例
            context_summary=context_summary,
            device_lookup_notes=device_facts,
        )
        final_message = HumanMessage(content=final_prompt)
        final_response = llm([final_message])
        final_response_text = (
            final_response.content
            if hasattr(final_response, "content")
            else str(final_response)
        )
        predicted_needs_tool, final_reasoning = parse_llm_response(
            final_response_text
        )

    is_correct = predicted_needs_tool == test_case.requires_human_interaction

    CONSOLE.log(
        f"[bold]预测是否需要 human_interaction_tool[/bold]: "
        f"{'需要' if predicted_needs_tool else '不需要'} "
        f"(期望: {'需要' if test_case.requires_human_interaction else '不需要'})"
    )
    CONSOLE.log("[bold]LLM 推理摘要[/bold]:")
    CONSOLE.log(final_reasoning)

    result: Dict[str, Any] = {
        "test_name": test_case.name,
        "user_command": test_case.user_command,
        "effective_user_name": effective_user_name,
        "types": test_case.types,
        "ground_truth": test_case.requires_human_interaction,
        "predicted": predicted_needs_tool,
        "is_correct": is_correct,
        "reasoning": final_reasoning,
        "llm_response": final_response_text,
        "chain_history": chain_history,
        "device_lookup_notes": device_facts,
    }

    # 将日志写入文件（按命令分类）
    if log_base_dir is not None:
        try:
            command_folder = _sanitize_filename(test_case.user_command)
            case_dir = log_base_dir.joinpath(command_folder)
            case_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            log_path = case_dir.joinpath(f"{test_case.name}_{timestamp}.json")

            log_content = {
                "test_name": test_case.name,
                "user_command": test_case.user_command,
                "effective_user_name": effective_user_name,
                "types": test_case.types,
                "ground_truth_requires_human_interaction": test_case.requires_human_interaction,
                "predicted_requires_human_interaction": predicted_needs_tool,
                "is_correct": is_correct,
                "preference_query": preference_query,
                "user_preferences": user_preferences,
                "device_state_summary": device_state_summary,
                "context_summary": context_summary,
                "reasoning": final_reasoning,
                "llm_response": final_response_text,
                "chain_history": chain_history,
                "device_lookup_notes": device_facts,
            }

            log_path.write_text(
                json.dumps(log_content, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            CONSOLE.log(f"[red]写入命令日志失败: {exc}")

    return result


def run_rag_cot_evaluation(config: RAGCOTConfig = None) -> Dict[str, Any]:
    """
    运行RAG COT链评估
    
    Args:
        config: RAG COT配置
    
    Returns:
        评估结果统计
    """
    if config is None:
        config = RAGCOTConfig()

    CONSOLE.log("[yellow]加载测试用例...")
    all_test_cases = load_test_cases()

    # 先剔除 google / test_set 类型的用例
    excluded_types = {"google", "test_set"}
    filtered_cases = [
        tc for tc in all_test_cases if not any(t in excluded_types for t in tc.types)
    ]

    # 根据配置进一步过滤测试用例
    if config.test_types_to_include:
        filtered_cases = [
            tc
            for tc in filtered_cases
            if any(t in config.test_types_to_include for t in tc.types)
        ]

    if config.max_test_cases is not None and config.max_test_cases > 0:
        filtered_cases = filtered_cases[: config.max_test_cases]
    
    CONSOLE.log(f"[green]共加载 {len(filtered_cases)} 个测试用例")
    
    # 创建本次评估的日志根目录
    root_dir = Path(os.getenv("SMARTHOME_ROOT", "."))
    rag_log_root = root_dir.joinpath("test_logs", "rag_cot")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_base_dir = rag_log_root.joinpath(timestamp)
    log_base_dir.mkdir(parents=True, exist_ok=True)
    CONSOLE.log(f"[yellow]RAG COT 日志将保存到: {log_base_dir}")

    # 实例化 LLM（禁用 streaming 以兼容同步调用）
    if hasattr(config.llm_config, "streaming") and config.llm_config.streaming:
        CONSOLE.log(
            "[yellow]检测到 LLM streaming=True，RAG COT 评估使用同步调用，现自动关闭 streaming。"
        )
        config.llm_config.streaming = False
    llm = config.llm_config.instantiate()

    # 初始化 UserProfileTool：通过 tools.py 封装的工具来检索用户偏好和历史
    CONSOLE.log("[yellow]初始化用户偏好检索工具（UserProfileTool）...")
    memory_path = f"{os.getenv('SMARTHOME_ROOT')}/data/memory_data/memory_bank.json"
    user_profile_config = UserProfileToolConfig(
        llm_config=config.llm_config,
        memory_path=memory_path,
    )
    user_profile_tool = user_profile_config.instantiate()

    context_tool = ContextUnderstandingTool()
    doc_root = os.getenv("SMARTHOME_ROOT", ".")
    docmanager_cache_path = Path(doc_root).joinpath(
        "external_api_docs", "cached_test_docmanager.json"
    )
    if not docmanager_cache_path.exists():
        raise FileNotFoundError(
            f"找不到 DocManager 缓存文件: {docmanager_cache_path}"
        )
    device_lookup_tool = DeviceLookupTool(
        docmanager_cache_path=docmanager_cache_path,
        max_results=config.device_lookup_max_results,
    )
    
    # 使用零样本（zero-shot）模式，不提供示例，确保评估公平性
    # 评估所有测试用例
    results = []
    correct_count = 0
    help_total = 0
    help_correct = 0
    non_help_total = 0
    non_help_correct = 0
    
    CONSOLE.log("[yellow]开始评估...")
    for i, test_case in enumerate(filtered_cases, 1):
        CONSOLE.log(f"[cyan]处理 {i}/{len(filtered_cases)}: {test_case.name}")
        
        device_lookup_tool.update_device_state(test_case.device_state)

        result = evaluate_test_case(
            test_case=test_case,
            llm=llm,
            user_profile_tool=user_profile_tool,
            context_tool=context_tool,
            device_lookup_tool=device_lookup_tool,
            preference_query_template=config.preference_query_template,
            user_name=config.user_name,  # 保留以保持兼容性，但实际会从命令中提取
            examples=None,  # 零样本模式，不提供示例
            log_base_dir=log_base_dir,
        )
        results.append(result)
        
        if result["is_correct"]:
            correct_count += 1
        else:
            CONSOLE.log(f"[red]错误: {test_case.name}")
            CONSOLE.log(f"  命令: {test_case.user_command}")
            CONSOLE.log(f"  期望: {'需要' if result['ground_truth'] else '不需要'}")
            CONSOLE.log(f"  预测: {'需要' if result['predicted'] else '不需要'}")

        # 统计“需要求助”与“不需要求助”场景下的正确率
        if result["ground_truth"]:
            help_total += 1
            if result["predicted"]:
                help_correct += 1
        else:
            non_help_total += 1
            if not result["predicted"]:
                non_help_correct += 1
    
    # 计算统计信息
    accuracy = correct_count / len(filtered_cases) if filtered_cases else 0
    help_accuracy = help_correct / help_total if help_total > 0 else 0
    non_help_accuracy = (
        non_help_correct / non_help_total if non_help_total > 0 else 0
    )
    
    # 按类型统计
    type_stats = {}
    for result in results:
        for test_type in result["types"]:
            if test_type not in type_stats:
                type_stats[test_type] = {"total": 0, "correct": 0}
            type_stats[test_type]["total"] += 1
            if result["is_correct"]:
                type_stats[test_type]["correct"] += 1
    
    summary = {
        "total_cases": len(filtered_cases),
        "correct": correct_count,
        "accuracy": accuracy,
        "help_total": help_total,
        "help_correct": help_correct,
        "help_accuracy": help_accuracy,
        "non_help_total": non_help_total,
        "non_help_correct": non_help_correct,
        "non_help_accuracy": non_help_accuracy,
        "type_statistics": {
            k: {
                "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0,
                "correct": v["correct"],
                "total": v["total"]
            }
            for k, v in type_stats.items()
        },
        "results": results
    }

    # 将最终汇总信息写入独立日志文件，便于外部查阅
    try:
        final_log_path = log_base_dir.joinpath("result_summary.json")
        final_log = {
            "generated_at": timestamp,
            "config": {
                "user_name": config.user_name,
                "max_test_cases": config.max_test_cases,
                "test_types_to_include": config.test_types_to_include,
                "device_lookup_max_results": config.device_lookup_max_results,
                "llm_model": getattr(config.llm_config, "model_name", "unknown"),
            },
            "summary": summary,
        }
        final_log_path.write_text(
            json.dumps(final_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        CONSOLE.log(f"[green]最终汇总日志已写入: {final_log_path}")
    except Exception as exc:
        CONSOLE.log(f"[red]写入最终汇总日志失败: {exc}")

    return summary


def print_evaluation_summary(summary: Dict[str, Any]):
    """打印评估结果摘要"""
    CONSOLE.rule("[bold green]评估结果摘要")
    CONSOLE.log(f"总测试用例数: {summary['total_cases']}")
    CONSOLE.log(f"正确预测数: {summary['correct']}")
    CONSOLE.log(f"准确率: {summary['accuracy']:.2%}")
    # 求助正确率：仅在“真实需要 human_interaction_tool”的样本上统计
    CONSOLE.log(
        f"求助正确率（在需要 human_interaction_tool 的样本中预测为需要的比例）: "
        f"{summary['help_accuracy']:.2%} "
        f"({summary['help_correct']}/{summary['help_total']})"
    )
    CONSOLE.log(
        f"不求助成功率（在不需要 human_interaction_tool 的样本中预测为不需要的比例）: "
        f"{summary['non_help_accuracy']:.2%} "
        f"({summary['non_help_correct']}/{summary['non_help_total']})"
    )
    
    CONSOLE.rule("[bold yellow]按类型统计")
    for test_type, stats in summary['type_statistics'].items():
        CONSOLE.log(
            f"{test_type}: {stats['correct']}/{stats['total']} "
            f"({stats['accuracy']:.2%})"
        )
    
    # 显示错误案例
    error_cases = [r for r in summary['results'] if not r['is_correct']]
    if error_cases:
        CONSOLE.rule("[bold red]错误案例")
        for case in error_cases[:10]:  # 最多显示10个错误案例
            CONSOLE.log(f"测试: {case['test_name']}")
            CONSOLE.log(f"  命令: {case['user_command']}")
            CONSOLE.log(f"  期望: {'需要' if case['ground_truth'] else '不需要'}")
            CONSOLE.log(f"  预测: {'需要' if case['predicted'] else '不需要'}")


if __name__ == "__main__":
    # 示例使用
    config = RAGCOTConfig(
        llm_config=GPTConfig(model_name="gpt-4o-mini", temperature=0.0)
    )
    
    summary = run_rag_cot_evaluation(config)
    print_evaluation_summary(summary)


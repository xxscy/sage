import csv
import inspect
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from types import SimpleNamespace

from langchain.schema.messages import HumanMessage
from langchain.utilities import OpenWeatherMapAPIWrapper

from sage.base import BaseConfig, GlobalConfig
from sage.coordinators.sage_coordinator import SAGECoordinatorConfig
from sage.smartthings.smartthings_tool import SmartThingsPlannerToolConfig
from sage.testing.testcases import TEST_CASE_TYPES, get_tests
from sage.utils.llm_utils import GPTConfig, LLMConfig
from sage.utils.common import CONSOLE
from sage.retrieval.tools import UserProfileToolConfig
from sage.testing.testing_utils import get_base_device_state
from sage.smartthings.docmanager import DocManager

# VLM 设备消歧工具（可选）
try:
    from sage.smartthings.device_disambiguation import VlmDeviceDetector
except Exception:
    VlmDeviceDetector = None

from sage.testing.rag_cot.testcase_loader import TestCaseInfo
from sage.testing.rag_cot.utils import (
    _summarize_device_state,
    build_tv_guide_knowledge,
    extract_user_name_from_command,
    _summarize_device_lookup_notes,
    _summarize_failure_notes,
)

def build_intent_analysis_prompt(
    test_case: TestCaseInfo,
    user_preferences: Optional[Dict[str, Any]] = None,
    user_memory_snippets: Optional[List[str]] = None,
    context_summary: Optional[str] = None,
    device_lookup_notes: Optional[List[str]] = None,
    environment_overview: Optional[str] = None,
    tv_guide_knowledge: Optional[str] = None,
    weather_reports: Optional[List[str]] = None,
    device_state_focus: Optional[str] = None,
    target_device_context: Optional[str] = None,
) -> str:
    """
    构建用户深层意图分析的 prompt，供 LLM 生成结构化意图描述。
    """
    preferences_text = str(user_preferences) if user_preferences else "(unavailable)"
    memory_text = "\n".join(user_memory_snippets) if user_memory_snippets else "(unavailable)"
    context_summary_text = context_summary or "(unavailable)"
    environment_text = environment_overview or "(unavailable)"
    device_state_focus_text = device_state_focus or "(unavailable)"
    target_device_context_text = target_device_context or "(unavailable)"
    tv_guide_text = tv_guide_knowledge or "(unavailable)"

    return f"""Analyze user intent and resolve ambiguities using device context and reasoning.

**Analysis Framework:**
1. **Intent Classification**: Device Control | Content Consumption | Information Query | Other
2. **Device Resolution**: Use device state and context to identify specific devices
   - Location-based: "by the TV", "in bedroom", "near credenza" → match with device locations
   - Type-based: "the light", "TV" → use most relevant/active device of that type
   - State-based: "turned off device", "dimmed lights" → match with current states
   - Multiple devices: prefer most recently used or most accessible
3. **Parameter Resolution**: Infer reasonable defaults and preferences
4. **Content Resolution**: Match with available content (TV channels, apps, media)

**Resolution Rules:**
- **Device Not Found**: Not an ambiguity - assume device lookup will handle this
- **Multiple Options**: Choose most reasonable based on context, location, recent usage
- **Unclear Parameters**: Use safe defaults or infer from preferences
- **Content Selection**: Choose best match from available options

**Risk Levels:**
- **Low**: Device control (lights, TV, audio, temperature), reversible actions
- **Medium**: Content selection, multi-device operations
- **High**: Security systems, irreversible actions, unknown devices

**Input:**
Command: "{test_case.user_command}"
Preferences: {preferences_text}
History: {memory_text}
TV Guide: {tv_guide_text}
Device State: {device_state_focus_text}
Device Context: {target_device_context_text}
Environment: {environment_text}
Context Summary: {context_summary_text}
Device Lookup: {chr(10).join(device_lookup_notes) if device_lookup_notes else "(none)"}

**Output Format:**
Refined Command: [clear, actionable command with resolved devices and parameters]
Risk Assessment: [Low/Medium/High - potential harm if executed incorrectly]
Confidence: [High/Low - certainty of interpretation and execution]"""


def build_environment_overview_prompt(
    *,
    test_case: TestCaseInfo,
    device_lookup_notes: Optional[List[str]] = None,
    device_state: Optional[Dict[str, Any]] = None,
    target_device_context: Optional[str] = None,
    vlm_disambiguation_notes: Optional[str] = None,
    image_hints: Optional[List[str]] = None,
) -> str:
    """
    Build a neutral environment summary (no reasoning, no inference).
    """
    lookup_text = "\n".join(device_lookup_notes) if device_lookup_notes else "(none)"
    device_state_text = _summarize_device_state(device_state)
    target_context_text = target_device_context or "(none)"
    vlm_notes_text = vlm_disambiguation_notes or "(none)"
    image_hints_text = "\n".join(image_hints) if image_hints else "(none)"

    return f"""Summarize environment facts only. No inference or speculation.

**Input:**
Command: "{test_case.user_command}"
Device Lookup: {lookup_text}
Device State: {device_state_text}
Target Devices: {target_context_text}
VLM Notes: {vlm_notes_text}
Image Hints: {image_hints_text}

**Output:** Two paragraphs:
1) Devices and locations (names, IDs, capabilities as stated)
2) Current states (power, levels, inputs, modes as provided)"""


def _extract_confidence_from_intent_analysis(intent_analysis: Optional[str]) -> str:
    """从意图分析文本中提取置信度（High/Low），默认返回Low"""
    if not intent_analysis:
        return "Low"
    # 查找 Confidence 字段（支持多种格式）
    confidence_patterns = [
        r"Confidence[：:]\s*(High|Low)",
        r"confidence[：:]\s*(High|Low)",
        r"Confidence[：:]\s*(High|Low|Medium)",
    ]
    for pattern in confidence_patterns:
        confidence_match = re.search(pattern, intent_analysis, re.IGNORECASE)
        if confidence_match:
            conf = confidence_match.group(1).capitalize()
            # Medium 视为 Low（需要更谨慎）
            return "High" if conf == "High" else "Low"
    return "Low"


def _extract_risk_assessment(intent_analysis: Optional[str]) -> str:
    """从意图分析文本中提取风险评估（Low/Medium/High），默认返回Medium"""
    if not intent_analysis:
        return "Medium"
    # 查找 Risk Assessment 字段
    risk_patterns = [
        r"Risk\s+Assessment[：:]\s*(Low|Medium|High)",
        r"risk\s+assessment[：:]\s*(Low|Medium|High)",
        r"Risk[：:]\s*(Low|Medium|High)",
    ]
    for pattern in risk_patterns:
        risk_match = re.search(pattern, intent_analysis, re.IGNORECASE)
        if risk_match:
            return risk_match.group(1).capitalize()
    return "Medium"


def build_cot_prompt(
    test_case: TestCaseInfo,
    user_preferences: Optional[Dict[str, Any]] = None,
    user_memory_snippets: Optional[List[str]] = None,
    examples: List[TestCaseInfo] = None,  # 保留参数以保持兼容性，但不再使用（零样本模式）
    context_summary: Optional[str] = None,
    device_lookup_notes: Optional[List[str]] = None,
    intent_analysis: Optional[str] = None,
    environment_overview: Optional[str] = None,
    tv_guide_knowledge: Optional[str] = None,
    weather_reports: Optional[List[str]] = None,
    device_state_focus: Optional[str] = None,
    target_device_context: Optional[str] = None,
    intent_reasoning_enabled: bool = True,
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
        intent_reasoning_enabled: 是否启用深度推理（受意图分析开关控制）

    Returns:
        构建好的 prompt 字符串（英文，用于发送给 LLM）
    """
    preferences_text = str(user_preferences) if user_preferences else "(no user preferences)"
    memory_text = "\n".join(user_memory_snippets) if user_memory_snippets else "(no history)"
    intent_analysis_text = intent_analysis if intent_analysis else "(unavailable)"
    environment_text = environment_overview if environment_overview else "(unavailable)"
    device_state_focus_text = device_state_focus if device_state_focus else "(unavailable)"
    
    # 提取置信度
    confidence = _extract_confidence_from_intent_analysis(intent_analysis)
    
    # 根据置信度选择不同的prompt策略
    if confidence == "High":
        return _build_high_confidence_prompt(test_case, intent_analysis_text, device_state_focus_text)
    else:
        return _build_low_confidence_prompt(test_case, intent_analysis_text, device_state_focus_text)


def _build_high_confidence_prompt(
    test_case: TestCaseInfo,
    intent_analysis_text: str,
    device_state_focus_text: str,
) -> str:
    """高置信度："""
    return f"""Decide if human_interaction_tool is needed. Focus on user clarification needs, not task execution feasibility.You only need to focus on the user's explicit instructions and not associate them with other requirements

**Rules:**
1. Information requests = Do not need
2. Content completely unavailable = Need
3. Safety/security critical decisions = Need
4. Pronoun ambiguity ("it/this/that") with multiple candidate devices (esp. same-type lights/TVs all on) = Need
5. All other device/parameter ambiguities = Do not need (assume resolution possible)
6. All other cases = Do not need


**Input:**
Command: "{test_case.user_command}"
Intent Analysis: {intent_analysis_text}
Device Context: {device_state_focus_text}

**Output:**
Reasoning: [Focus on clarification needs only]
Conclusion: Need / Do not need
Reason: [Only if user clarification is truly needed]"""


def _build_low_confidence_prompt(
    test_case: TestCaseInfo,
    intent_analysis_text: str,
    device_state_focus_text: str,
) -> str:
    """低置信度："""
    return f"""Decide if human_interaction_tool is needed. Only care about user clarification requirements.You only need to focus on the user's explicit instructions and not associate them with other requirements.

**Rules:**
1. Information requests = Do not need
2. Content completely unavailable with no alternatives = Need
3. Safety/security critical decisions requiring user judgment = Need
4. Subjective preferences with no user history/context = Need
5. Pronoun ambiguity ("it/this/that") with multiple candidate devices (esp. same-type lights/TVs all on) = Need
6. When a user's instruction explicitly requires specifying a certain device, but this device is not found = Need
7. All device/parameter resolution issues = Do not need (assume can be handled)
8. All other cases = Do not need

**Input:**
Command: "{test_case.user_command}"
Intent Analysis: {intent_analysis_text}
Device Context: {device_state_focus_text}

**Output:**
Reasoning: [Focus only on clarification needs]
Conclusion: Need / Do not need
Reason: [Only if user input is essential]"""


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
        "(none)"
        if not chain_history
        else "\n".join([f"Step {s['step']}: {s['action']} - {s['reasoning']}" for s in chain_history])
    )

    preference_status = "available" if context_state.get("user_preferences") else "missing"
    summary_status = "available" if context_state.get("context_summary") else "missing"
    device_facts_list = context_state.get("device_facts") or []
    device_fact_status = "available" if device_facts_list else "missing"
    device_facts_detail = _summarize_device_lookup_notes(device_facts_list)
    weather_status = "available" if context_state.get("weather_facts") else "missing"
    device_state_focus_text = context_state.get("device_state_focus") or "(unavailable)"
    failure_notes = context_state.get("failure_notes") or []
    failure_notes_summary = _summarize_failure_notes(failure_notes)

    return f"""Select the next tool to gather information. Do not infer final answer.

**Command:** "{test_case.user_command}"

**Status:**
Preferences: {preference_status}
Device Facts: {device_fact_status}
{device_facts_detail}
Context Summary: {summary_status}
Weather: {weather_status}
Device State: {device_state_focus_text}
Failures: {failure_notes_summary}

**Tools:**
1. preference_lookup: {{"query": "<english>"}} (default: "{preference_query_template}")
2. device_lookup: {{"query": "<device/action>"}}
3. weather_lookup: {{"query": "<City, Country>"}}
4. context_summary: {{}}
5. final_decision: {{}} (when sufficient or no more info available)

**Output (JSON only):**
{{"thought": "<reason>", "action": "<tool_name>", "query": "<optional>"}}"""
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
    LLM 需要分析命令并判断是否有足够信息执行，输出结构化结果。
    """
    preferences_text = str(user_preferences) if user_preferences else "(unavailable)"
    memory_text = "\n".join(user_memory_snippets) if user_memory_snippets else "(unavailable)"
    context_summary_text = context_summary or "(unavailable)"
    environment_text = environment_overview or "(unavailable)"
    device_state_focus_text = device_state_focus or "(unavailable)"
    target_device_context_text = target_device_context or "(unavailable)"
    tv_guide_text = tv_guide_knowledge or "(unavailable)"

    return f"""# Role
You are an intent analysis module for a smart home assistant.

# Task
Analyze the command and determine if execution is possible with available information.

# Decision Process:
Examine the command and available context. Prefer saying "NO" (no clarification needed) when information is sufficient or can be reasonably inferred.

**Say "NO" (no clarification needed) when:**
- Device is identified through spatial references (location relative to objects or spaces) or explicit naming.
- Scope is unambiguous (applies to all relevant items or a single unique target).
- Intent can be inferred from purpose or context clues (commands describing desired state or purpose).
- Parameters can use sensible defaults (adjustment verbs without specific values).
- Requested information is available in the provided context.
- Only one relevant device exists or is active.

**Say "YES" (clarification needed) only when ALL conditions are met for critical information:**
- Device reference is ambiguous (pure pronoun without context) AND multiple candidate devices are active.
- Personal preference is explicitly referenced BUT preference data is unavailable.
- Specific content is required (definite article with category) BUT preferences or context cannot determine which.
- Term is unclear or undefined AND no context explains its meaning.

# Input
Command: "{test_case.user_command}"
User Preferences: {preferences_text}
Interaction History: {memory_text}
TV Guide: {tv_guide_text}
Device State: {device_state_focus_text}
Device Context: {target_device_context_text}
Environment: {environment_text}
Context Summary: {context_summary_text}
Device Lookup: {chr(10).join(device_lookup_notes) if device_lookup_notes else "(none)"}

# Output Format
Refined Command: [Specific actionable command - only if NO clarification needed]
Needs Clarification: [YES/NO]
Missing Information: [If YES: what is missing; if NO: "None"]
Confidence: [High/Low]
Risk Assessment: [Low/Medium/High]"""


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


def _extract_needs_clarification(intent_analysis: Optional[str]) -> bool:
    """从 Intent Analysis 中提取是否需要澄清（由 LLM 判断）
    
    Returns: True if LLM determined clarification is needed
    """
    if not intent_analysis:
        return False
    
    lower_text = intent_analysis.lower()
    
    # 检测 "Needs Clarification: YES"
    if "needs clarification: yes" in lower_text or "needs clarification:** yes" in lower_text:
        return True
    
    return False


def _extract_missing_information(intent_analysis: Optional[str]) -> Optional[str]:
    """从 Intent Analysis 中提取缺失的信息（由 LLM 判断）
    
    Returns: Missing information string, or None if not found
    """
    if not intent_analysis:
        return None
    
    # 查找 Missing Information 字段
    patterns = [
        r"Missing Information[：:]\s*(.+?)(?:\n|$)",
        r"Missing Information[：:]\*\*\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, intent_analysis, re.IGNORECASE)
        if match:
            info = match.group(1).strip()
            if info.lower() not in ["none", "n/a", ""]:
                return info
    return None


def _extract_refined_command(intent_analysis: Optional[str]) -> Optional[str]:
    """从 Intent Analysis 中提取 Refined Command"""
    if not intent_analysis:
        return None
    
    patterns = [
        r"Refined Command[：:]\s*(.+?)(?:\n|$)",
        r"Refined Command[：:]\*\*\s*(.+?)(?:\n|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, intent_analysis, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


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
    effective_user_name: Optional[str] = None,
) -> str:
    """
    构建 COT（Chain of Thought）推理提示词。
    所有判断都由 LLM 完成，代码只负责传递信息。
    """
    # 从 Intent Analysis 提取 LLM 的判断结果
    needs_clarification = _extract_needs_clarification(intent_analysis)
    missing_info = _extract_missing_information(intent_analysis)
    refined_command = _extract_refined_command(intent_analysis)
    
    # 构建上下文摘要（只传递信息，不做判断）
    context_parts = []
    
    # 1. Intent Analysis 结果（完整显示，让最终决策 LLM 自己判断）
    if intent_analysis:
        context_parts.append(f"**Intent Analysis Result:**\n{intent_analysis}")
    
    # 2. 用户信息
    context_parts.append(f"**User:** {effective_user_name or 'unknown'}")
    
    # 3. 用户偏好（原样传递，让 LLM 自己判断是否足够）
    if user_preferences:
        context_parts.append(f"**User Preferences:** {user_preferences}")
    else:
        context_parts.append("**User Preferences:** (not available)")
    
    # 4. 设备和环境信息（原样传递）
    if environment_overview:
        context_parts.append(f"**Environment:** {environment_overview}")
    if device_state_focus:
        context_parts.append(f"**Device State:** {device_state_focus}")
    if target_device_context:
        context_parts.append(f"**Device Context:** {target_device_context}")
    if device_lookup_notes:
        context_parts.append(f"**Available Devices:**\n{chr(10).join(device_lookup_notes)}")
    if context_summary:
        context_parts.append(f"**Context Summary:** {context_summary}")
    
    context_text = "\n\n".join(context_parts) if context_parts else "(unavailable)"
    
    return _build_voi_decision_prompt(test_case, context_text)


def _build_voi_decision_prompt(
    test_case: TestCaseInfo,
    context_text: str,
) -> str:
    """最终决策 prompt - 完全由 LLM 自主判断，无硬编码规则"""
    
    return f"""# Role
You are SAGE, a smart home assistant.

# Task
Decide: **ACT** (execute directly) or **ASK** (clarify first).

# Decision Process:
Make a decision based on the clarity of the command and available context. Prefer ACT when information is sufficient or can be reasonably inferred.

**Important:** If Intent Analysis indicates "Needs Clarification: NO", you should ACT unless there is a clear reason not to.

**ACT (execute directly) when:**
- Device or target is identified through spatial references (location relative to objects or spaces) or explicit naming.
- Scope is unambiguous (applies to all relevant items or a single unique target).
- Intent can be inferred from purpose or context clues (commands describing desired state or purpose).
- Parameters can use sensible defaults (adjustment verbs without specific values).
- Necessary information is available in the provided context (queries about state, value, or condition with available data).

**ASK (clarify first) only when ALL conditions are met for critical information:**
- Device reference is ambiguous (pure pronoun without context) AND multiple candidate devices are active.
- Personal preference is explicitly referenced BUT preference data is unavailable.
- Specific content is required (definite article with category) BUT preferences or context cannot determine which.
- Term is unclear or undefined AND no context explains its meaning.

# Context
**Original Request:** "{test_case.user_command}"

{context_text}

# Output
Thought: [Analyze the command and context. If Intent Analysis says "Needs Clarification: NO", you should ACT unless there is a clear reason not to. Prefer ACT when information is sufficient or can be reasonably inferred.]
Decision: [ACT / ASK]
Question: [If ASK, your clarifying question; if ACT, leave empty]"""


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
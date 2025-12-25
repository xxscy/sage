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
    preferences_text = str(user_preferences) if user_preferences else "(no user preference information available)"
    memory_text = "\n".join(user_memory_snippets) if user_memory_snippets else "(no historical interactions)"
    context_summary_text = context_summary or "(context summary unavailable)"
    environment_text = environment_overview or "(environment overview unavailable)"
    device_state_focus_text = device_state_focus or "(device state focus unavailable)"
    target_device_context_text = target_device_context or "(targeted device context unavailable)"
    tv_guide_text = tv_guide_knowledge or "(tv guide knowledge unavailable)"

    return f"""You are the "Deep Intent Resolver".

Your task is to translate the user's raw input into a precise "Refined Command" using strict logical deduction.

**Reasoning Framework (Zero-Shot Logic):**

1. **Categorize the Request**:
   - **Device Control**: Physical state changes. Missing parameters may be inferred safely from context and use conservative defaults.
   - **Content Consumption**: Playing specific media. A verified source is required; do not assume a source when missing.

2. **Content Verification Protocol (mandatory for content requests)**:
   - If no channel/app is given, search `TV Guide` or history for a semantic match.
   - Do not assume the current input/channel is correct just because the device is on.
   - If a verifiable source is found, specify the exact channel/input in the Refined Command; if not, mark UNKNOWN_CHANNEL or AMBIGUOUS_SOURCE—do not guess.

3. **Event-State Causality (device control)**:
   - Use user activity/state to decide environment changes: reduce interference for focus/rest; increase illumination for visibility.

4. **Entity Grounding**:
   - Ground targets by location/name; when multiple similar devices exist, prefer the single available/active one to remove ambiguity.
   - **CRITICAL: Collective Reference Recognition**: If the command uses collective references (e.g., "all", "every", "entire", "both", "all the", plural nouns like "lights" when referring to multiple devices), this indicates a group target, NOT ambiguity. Do NOT mark as AMBIGUOUS_DEVICE. The command should target all matching devices in the group.
   - **CRITICAL: Singular Reference Constraint**: If the command uses a singular reference (pronoun "it", "this", "that", or possessive "my X" implying a single target) AND multiple candidate devices exist without a clear, unambiguous single match (e.g., VLM ambiguous, multiple active devices of same type), you MUST mark the target as **AMBIGUOUS_DEVICE** in the Refined Command. Do NOT arbitrarily select a "first instance" or "active one" when the command implies a single target but multiple candidates exist.
   - **Context-Based Disambiguation**: Use provided context (user preferences, device state, location hints) to resolve ambiguity. If context clearly identifies a device (e.g., "same room as TV", "by the plant", user preference rules), use that context to ground the device. Only mark as AMBIGUOUS_DEVICE if context cannot resolve the ambiguity.

**Preference Handling Rules (CRITICAL):**
- **Explicit preference references**: If the user explicitly references a personal preference (e.g., "my favorite X", "my usual Y", "my preferred Z"), you MUST verify this preference is retrievable from the provided preferences/history.
  - If the preference is **found and verified** in the provided data -> Use it in the Refined Command.
  - If the preference is **not found or not retrievable** -> Mark as `UNKNOWN_PARAMETER` in the Refined Command. **Do NOT infer or guess** the preference from unrelated context or general knowledge.
  - **Rationale**: Personal preferences are user-specific and cannot be safely inferred; guessing wrong preferences violates user trust.

**Anti-Hallucination Guard (CRITICAL):**
- Ground every decision **only** on the provided facts (preferences, history, device lookup, device state, guide/knowledge). If a required fact is absent or ambiguous, mark it as `UNKNOWN_PARAMETER` rather than inventing it.
- Preference usage must be **explicitly supported** by the provided data for the same user and relevant device/context. Generic statements or weak associations do **not** authorize assuming a preference.
- If multiple candidate targets or preferences exist without a clear single winner, treat the target/preference as **ambiguous** and mark as `AMBIGUOUS_DEVICE` or `UNKNOWN_PARAMETER` in the Refined Command.
- When command uses singular references (pronouns, "it", "this", "that", "my X") and multiple devices fit, you MUST treat the target as **AMBIGUOUS_DEVICE** and mark it in the Refined Command. Do NOT arbitrarily select one device.
- **Subjective/Qualitative Descriptors**: If the user uses subjective or qualitative descriptors (e.g., "redonkulous", "cozy", "vibrant", "comfortable"), check if user preferences/history provide mappings for these terms. If a mapping exists (e.g., "cozy" -> dim warm light in preferences), use it. Only mark as `UNKNOWN_PARAMETER` if no such mapping exists in the provided data.

**Execution Bias Guidelines (apply when not conflicting above):**
- Align actions with the user's expressed comfort/goal; adjust intensity downward when user indicates overload, upward when user indicates insufficiency.
- When direction is known but parameters are missing, apply conservative defaults; mark UNKNOWN when:
  - The user asked for a specific-but-unknown preference (highest priority)
  - No safe default can be inferred from context, device capabilities, or general patterns
  - The user uses subjective/qualitative descriptors that cannot be objectively mapped from provided preferences/history (e.g., "redonkulous" with no preference context)
- **Subjective Descriptor Mapping**: If user preferences/history provide mappings for subjective descriptors (e.g., "cozy" -> dim warm light, "vibrant" -> bright color), use those mappings. Only mark as UNKNOWN_PARAMETER if no such mapping exists in provided data.
- If VLM/lookup or naming/location yields a high-confidence single target, use it; similar devices do not block execution. **However**, if VLM indicates ambiguity or multiple candidates exist for a singular reference, mark as `AMBIGUOUS_DEVICE`.
- When preferences/history specify an entity or mode but omit an attribute, infer the typical attribute and label it as inferred; leave unknown only when no reasonable basis exists. **However**, do NOT infer from subjective descriptors or when the user explicitly requests a personal preference that is not retrievable.
- **Context Utilization**: Actively use provided context (user preferences, device state, location hints) to resolve ambiguity. If context provides a clear rule (e.g., "same room as TV", "by the plant"), use it to ground devices without marking as AMBIGUOUS_DEVICE.

**Task:**
Generate a `Refined Command`.
- For **Device Control**: 
  - Apply safe defaults if parameters are missing AND the user did not explicitly request a personal preference.
  - If user explicitly requested a personal preference that is not retrievable -> Mark as `UNKNOWN_PARAMETER`.
- For **Content Consumption**: Only generate a specific command if the source is **verified**. Otherwise, flag the missing parameter.

**Facts:**
- Command: "{test_case.user_command}"
- TV Guide / Knowledge: {tv_guide_text}
- User History: {memory_text}
- Device State Focus: {device_state_focus_text}
- Target Device Context: {target_device_context_text}
- Context Summary: {context_summary_text}
- Device Environment: {environment_text}
- Preferences / device facts: {preferences_text}
- Device Lookup Results:
{chr(10).join(device_lookup_notes) if device_lookup_notes else "(no device lookup performed)"}


"""


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
    lookup_text = (
        "\n".join(device_lookup_notes)
        if device_lookup_notes
        else "(device lookup not used)"
    )
    device_state_text = _summarize_device_state(device_state)
    target_context_text = (
        target_device_context
        if target_device_context
        else "(no targeted device context generated)"
    )
    vlm_notes_text = (
        vlm_disambiguation_notes
        if vlm_disambiguation_notes
        else "(vlm disambiguation not used)"
    )
    image_hints_text = (
        "\n".join(image_hints) if image_hints else "(no image hints gathered)"
    )

    return f"""You are compiling an environment briefing for a smart home assistant.
Summarize only the known facts; do NOT infer or speculate. Keep it concise, in English.

Facts to include:
- User command: "{test_case.user_command}"
- Device lookup notes:
{lookup_text}
- Device state snapshot (from fake_requests): {device_state_text}
- Target devices extracted from the command:
{target_context_text}
- VLM disambiguation hint:
{vlm_notes_text}
- Image hints (device_id -> image path):
{image_hints_text}

Output exactly 2 short paragraphs:
1) Devices & locations (as stated or hinted by names/metadata); list device names/IDs and capabilities if present.
2) Current states or control cues (power, levels, input sources, modes) exactly as provided.

Do not add interpretations, guesses, or recommendations."""


def _extract_confidence_from_intent_analysis(intent_analysis: Optional[str]) -> str:
    """从意图分析文本中提取置信度（High/Low），默认返回Low"""
    if not intent_analysis:
        return "Low"
    # 查找 Confidence 字段
    confidence_match = re.search(
        r"Confidence[：:]\s*(High|Low)",
        intent_analysis,
        re.IGNORECASE,
    )
    if confidence_match:
        return confidence_match.group(1).capitalize()
    return "Low"


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
    """高置信度prompt：简洁，鼓励执行"""
    return f"""You are the decision module of a smart home assistant. Complete user objectives efficiently.

**Core Principle:** Execute user commands. Prefer action over inquiry when information is sufficient.

**Decision Rules:**
1. **Information requests**: If command seeks to retrieve/report information -> **Do not need**.
2. **Check markers**: If `Refined Command` contains `AMBIGUOUS_DEVICE` or `UNKNOWN_PARAMETER` -> **Need**.
3. **Content requests**: If Content Consumption and source is verified -> **Do not need**; if unverified -> **Need**.
4. **Device control**: If device and parameters are specified (no markers) -> **Do not need**. Apply safe defaults when direction is clear.

**Input:**
- Command: "{test_case.user_command}"
- Intent Analysis:
{intent_analysis_text}
- Device Context:
{device_state_focus_text}

**Process:**
1. Check if information request -> If yes, **Do not need**.
2. Check for `AMBIGUOUS_DEVICE` or `UNKNOWN_PARAMETER` in `Refined Command` -> If found, **Need**.
3. Classify intent (Device Control / Content Consumption).
4. For Content: if source verified -> **Do not need**; else -> **Need**.
5. For Device Control: if markers absent and device/parameters specified -> **Do not need**.

**Output Format:**
Reasoning: Step 1 - [...] Step 2 - [...] Step 3 - [...] Step 4 - [...] Step 5 - [...]

Conclusion:
Need / Do not need to use human_interaction_tool

Reason: [Brief explanation]"""


def _build_low_confidence_prompt(
    test_case: TestCaseInfo,
    intent_analysis_text: str,
    device_state_focus_text: str,
) -> str:
    """低置信度prompt：谨慎但允许上下文消歧"""
    return f"""You are the decision module of a smart home assistant. Exercise caution but utilize context when available.

**Core Principle:** When confidence is low, prioritize accuracy. However, if context (preferences, device state, location) provides sufficient information to resolve ambiguity, proceed without asking.

**Critical Checks (apply in order):**
1. **Information requests**: If command retrieves/reports information -> **Do not need**.
2. **Collective references**: If command uses "all", "every", "entire", "both", or plural nouns referring to groups -> **Do not need** (group target, not ambiguity).
3. **Context-based resolution**: If `Refined Command` contains markers BUT context (user preferences, device state, location hints) clearly resolves the ambiguity -> **Do not need**.
4. **Mandatory markers without context**: If `Refined Command` contains `AMBIGUOUS_DEVICE` or `UNKNOWN_PARAMETER` AND context cannot resolve -> **Need**.
5. **Content verification**: If Content Consumption and source is unverified -> **Need**.
6. **Subjective descriptors**: If subjective descriptor (e.g., "cozy", "vibrant") can be mapped from user preferences -> **Do not need**; if cannot be mapped -> **Need**.

**Input:**
- Command: "{test_case.user_command}"
- Intent Analysis:
{intent_analysis_text}
- Device Context:
{device_state_focus_text}

**Process:**
1. Check if information request -> If yes, **Do not need**.
2. Check if collective reference ("all", "every", "entire", plural) -> If yes, **Do not need**.
3. Check `Refined Command` for markers -> If present, evaluate if context resolves ambiguity:
   - If user preferences/device state/location clearly identify device -> **Do not need**
   - If context cannot resolve -> **Need**
4. For Content: if source verified -> **Do not need**; else -> **Need**.
5. For subjective descriptors: if mappable from preferences -> **Do not need**; else -> **Need**.
6. Final verdict: Use context when available; ask only when truly ambiguous.

**Output Format:**
Reasoning: Step 1 - [...] Step 2 - [Check collective/context] Step 3 - [Evaluate markers with context] Step 4 - [...] Step 5 - [...] Step 6 - [...]

Conclusion:
Need / Do not need to use human_interaction_tool

Reason: [Explanation of ambiguity status and context utilization]"""


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

    device_facts_list = context_state.get("device_facts") or []
    device_fact_status = "available" if device_facts_list else "missing"
    device_facts_detail = _summarize_device_lookup_notes(device_facts_list)
    indented_device_facts = "\n".join(
        f"  {line}" for line in device_facts_detail.splitlines()
    )
    weather_status = (
        "available" if context_state.get("weather_facts") else "missing"
    )
    device_state_focus_text = (
        context_state.get("device_state_focus")
        if context_state.get("device_state_focus")
        else "(device state focus unavailable)"
    )
    failure_notes = context_state.get("failure_notes") or []
    failure_notes_summary = _summarize_failure_notes(failure_notes)

    planner_prompt = f"""You are a planner whose ONLY job is to pick the next tool to gather information relevant to the user command.
Do NOT infer the final answer and do NOT judge safety. Just gather info; then hand off to final_decision.

Goal: quickly collect only the necessary info related to the command (preferences / device facts / weather when relevant). Avoid redundant or irrelevant lookups (e.g., skip weather if the request is unrelated).

Current user command: "{test_case.user_command}"

Progress so far:
- User preference / device facts: {preference_status}
- Device lookup insights: {device_fact_status}
  Latest samples:
{indented_device_facts}
- Context summary: {summary_status}
- Weather lookup insights: {weather_status}
- Device state focus: {device_state_focus_text}
- Retrieval issues encountered:
{failure_notes_summary}

Available tools (pick exactly one per step):
1) preference_lookup -> {{"query": "<english query>"}}
   If query is omitted, use template: "{preference_query_template}"
2) device_lookup     -> {{"query": "<english device/action description>"}}
3) weather_lookup    -> {{"query": "<City, Country>"}}
   If omitted, use the default location
4) context_summary   -> no params, produce latest consolidated summary
5) final_decision    -> no params; call when you think information is sufficient or no more can be gained

Output strict JSON only:
{{
  "thought": "<one short reason for the tool choice>",
  "action": "<preference_lookup | device_lookup | weather_lookup | context_summary | final_decision>",
  "query": "<optional query for preference/device/weather>"
}}
Do NOT add any other text."""
    return planner_prompt
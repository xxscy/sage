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
            "weather_lookup",
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
    
    支持两种格式：
    1. 新的 VoI 格式：Decision: [ACT / ASK]
    2. 旧的格式：Conclusion: Need / Do not need
    
    Returns:
        (是否需要human_interaction_tool, 推理过程)
    """
    response_lower = response.lower()
    
    # 优先解析新的 VoI 格式：Decision: ACT / ASK
    decision_patterns = [
        r'Decision[：:]\s*(ACT|ASK)',
        r'\*{0,2}Decision:\*{0,2}\s*(ACT|ASK)',
    ]
    for pattern in decision_patterns:
        decision_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if decision_match:
            decision_text = decision_match.group(1).upper().strip()
            if decision_text == "ASK":
                return True, response
            elif decision_text == "ACT":
                return False, response
    
    # 兼容旧的格式：Conclusion: Need / Do not need
    # 查找中文或英文结论部分（支持加粗标记 **Conclusion:** 或 Conclusion:）
    conclusion_patterns = [
        # 中文格式：结论：需要/不需要（可能有加粗标记 **）
        r'\*{0,2}结论[：:]\s*(需要|不需要)',
        # 英文格式：**Conclusion:** Need 或 Conclusion: Need（支持加粗标记）
        r'\*{0,2}Conclusion:\*{0,2}\s*(Need|Do\s*not\s*need)\s*',
        # 英文格式：Conclusion: Need/Do not need（忽略大小写，可能有加粗标记）
        r'\*{0,2}[Cc]onclusion:\*{0,2}\s*(Need|Do\s*not\s*need)\s*',
    ]
    
    for pattern in conclusion_patterns:
        conclusion_match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if conclusion_match:
            # 提取匹配的文本
            matched_text = conclusion_match.group(1).lower().strip()
            # 判断是否需要工具
            if matched_text in ["需要", "need"]:
                return True, response
            elif matched_text in ["不需要", "do not need", "don't need"]:
                return False, response
    
    # 如果没有找到明确的结论格式，尝试关键词匹配（兼容中英文）
    # 查找 "Need" 或 "Do not need" 在结论附近
    need_patterns = [
        r'conclusion[：:]\s*need\b',  # Conclusion: Need
        r'conclusion[：:]\s*do\s+not\s+need',  # Conclusion: Do not need
    ]
    for pattern in need_patterns:
        match = re.search(pattern, response_lower, re.IGNORECASE)
        if match:
            if "do not need" in match.group(0) or "don't need" in match.group(0):
                return False, response
            else:
                return True, response
    
    # 查找独立的 "Need" 或 "Do not need"（在结论行）
    if re.search(r'\bneed\s+to\s+use\s+human', response_lower):
        return True, response
    if re.search(r'do\s+not\s+need\s+to\s+use\s+human', response_lower):
        return False, response
    if re.search(r"don'?t\s+need\s+to\s+use\s+human", response_lower):
        return False, response
    
    # 中文关键词匹配
    if "需要" in response and "human_interaction" in response_lower:
        if "不需要" not in response[:response.find("需要")]:
            return True, response
    if "不需要" in response or "不需要使用" in response:
        return False, response
    
    # 默认返回False
    return False, response

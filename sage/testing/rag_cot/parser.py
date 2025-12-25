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

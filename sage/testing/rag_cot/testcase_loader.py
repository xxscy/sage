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

from sage.testing.rag_cot.utils import _InitialStateCapture

# 这些函数已移到utils.py中，从utils导入


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
        self._prepared_state: Optional[Dict[str, Any]] = None


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

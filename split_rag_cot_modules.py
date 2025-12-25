"""
自动拆分 rag_cot_chain.py 为多个模块
"""
import re
from pathlib import Path

source_file = Path("sage/testing/rag_cot_chain.py")
output_dir = Path("sage/testing/rag_cot")

# 读取源文件
with open(source_file, "r", encoding="utf-8") as f:
    content = f.read()
    lines = content.splitlines()

# 通用导入部分
common_imports = """import csv
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

TV_GUIDE_PATH = Path(__file__).parent.parent.joinpath("tv_guide.csv")
_TV_GUIDE_CACHE: Optional[List[Dict[str, str]]] = None
"""

# 提取行号范围
def extract_lines(start, end):
    """提取行号范围（1-based）"""
    return "\n".join(lines[start-1:end])

# 1. tools.py - 工具类
tools_start = 73
tools_end = 538
tools_code = extract_lines(tools_start, tools_end)
tools_content = common_imports + "\n" + f"""
from sage.testing.rag_cot.utils import (
    _ensure_tool_global_config,
    _vlm_pick_device,
)

""" + tools_code

with open(output_dir / "tools.py", "w", encoding="utf-8") as f:
    f.write(tools_content)
print("[OK] Created tools.py")

# 2. utils.py - 辅助函数
utils_functions = [
    (430, 510),  # _is_device_lookup_failure 到 _is_weather_lookup_failure
    (540, 1200),  # _InitialStateCapture 到 _ensure_tool_global_config
    (1057, 1200),  # VLM相关函数
]

utils_code_parts = []
for start, end in utils_functions:
    utils_code_parts.append(extract_lines(start, end))

utils_content = common_imports + "\n" + "\n\n".join(utils_code_parts)

with open(output_dir / "utils.py", "w", encoding="utf-8") as f:
    f.write(utils_content)
print("[OK] Created utils.py")

# 3. testcase_loader.py
testcase_start = 1225
testcase_end = 1574
testcase_code = extract_lines(testcase_start, testcase_end)
testcase_content = common_imports + "\n" + f"""
from sage.testing.rag_cot.utils import (
    _prepare_device_state_for_test,
    _get_state_capture_test_config,
)

""" + testcase_code

with open(output_dir / "testcase_loader.py", "w", encoding="utf-8") as f:
    f.write(testcase_content)
print("[OK] Created testcase_loader.py")

# 4. prompts.py
prompts_start = 1576
prompts_end = 1950
prompts_code = extract_lines(prompts_start, prompts_end)
prompts_content = common_imports + "\n" + f"""
from sage.testing.rag_cot.testcase_loader import TestCaseInfo
from sage.testing.rag_cot.utils import _summarize_device_state

""" + prompts_code

with open(output_dir / "prompts.py", "w", encoding="utf-8") as f:
    f.write(prompts_content)
print("[OK] Created prompts.py")

# 5. parser.py
parser_start = 1953
parser_end = 2024
parser_code = extract_lines(parser_start, parser_end)
parser_content = common_imports + "\n" + parser_code

with open(output_dir / "parser.py", "w", encoding="utf-8") as f:
    f.write(parser_content)
print("[OK] Created parser.py")

# 6. evaluator.py
evaluator_start = 2026
evaluator_end = 2795
evaluator_code = extract_lines(evaluator_start, evaluator_end)
evaluator_content = common_imports + "\n" + f"""
from sage.testing.rag_cot.config import RAGCOTConfig
from sage.testing.rag_cot.tools import (
    ContextUnderstandingTool,
    DeviceLookupTool,
    WeatherLookupTool,
)
from sage.testing.rag_cot.testcase_loader import TestCaseInfo, load_test_cases
from sage.testing.rag_cot.prompts import (
    build_intent_analysis_prompt,
    build_environment_overview_prompt,
    build_cot_prompt,
    build_chain_planner_prompt,
    build_tv_guide_knowledge,
    extract_user_name_from_command,
)
from sage.testing.rag_cot.parser import parse_planner_response, parse_llm_response
from sage.testing.rag_cot.utils import (
    _sanitize_filename,
    _prepare_device_state_for_test,
    _vlm_pick_device,
    _has_device_image,
    _collect_image_hints,
    _collect_vlm_candidates,
    _detect_target_device_categories,
    _infer_device_categories_from_metadata,
    _is_device_lookup_failure,
    _is_weather_lookup_failure,
    _summarize_device_lookup_notes,
    _summarize_failure_notes,
    _serialize_for_compare,
    _shorten_text,
    _normalize_query_text,
    _record_failure_note,
    _should_skip_query,
)

""" + evaluator_code

with open(output_dir / "evaluator.py", "w", encoding="utf-8") as f:
    f.write(evaluator_content)
print("[OK] Created evaluator.py")

print("\n所有模块已创建完成！")


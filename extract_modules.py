"""
自动提取模块代码的脚本
"""
from pathlib import Path

source_file = Path("sage/testing/rag_cot_chain.py")
with open(source_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 定义各个模块的行号范围（基于实际代码结构）
modules_info = {
    "tools": {
        "start": 73,  # ContextUnderstandingTool
        "end": 538,   # WeatherLookupTool结束
        "imports": """import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.utilities import OpenWeatherMapAPIWrapper

from sage.base import BaseConfig, GlobalConfig
from sage.smartthings.smartthings_tool import SmartThingsPlannerToolConfig
from sage.utils.llm_utils import LLMConfig
from sage.utils.common import CONSOLE
from sage.smartthings.docmanager import DocManager

# VLM 设备消歧工具（可选）
try:
    from sage.smartthings.device_disambiguation import VlmDeviceDetector
except Exception:
    VlmDeviceDetector = None

from sage.testing.rag_cot.utils import (
    _ensure_tool_global_config,
    _vlm_pick_device,
)
"""
    },
}

def extract_lines(start_line, end_line):
    """提取指定行号范围的代码（1-based）"""
    return "".join(lines[start_line-1:end_line])

# 提取tools模块
tools_code = extract_lines(73, 538)
tools_content = modules_info["tools"]["imports"] + "\n" + tools_code

# 写入文件
output_dir = Path("sage/testing/rag_cot")
tools_file = output_dir / "tools.py"
with open(tools_file, "w", encoding="utf-8") as f:
    f.write(tools_content)

print(f"Created {tools_file}")












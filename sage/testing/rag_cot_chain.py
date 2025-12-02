"""
RAG 系统的 COT 链，用于判断是否需要使用 human_interaction_tool 工具。

该模块从 testcases.py 中读取测试用例，先根据用户指令构造检索请求，
检索与该用户相关的偏好信息和设备信息，然后在包含这些上下文的
中文 COT prompt 中，让 LLM 判断是否需要使用 human_interaction_tool
来澄清用户问题。
"""
import csv
import inspect
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from types import SimpleNamespace
TV_GUIDE_PATH = Path(__file__).with_name("tv_guide.csv")
_TV_GUIDE_CACHE: Optional[List[Dict[str, str]]] = None


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
        "You are retrieving stored preferences for the smart-home assistant. "
        "Return ONLY factual preference statements already in memory related to this command: "
        "{user_command}\n"
        "- Do NOT infer, guess, or ask the user anything.\n"
        "- Do NOT include suggestions, plans, or follow-up questions.\n"
        "- If no direct preference exists, reply with an explicit '(no preference found)'."
    )
    max_test_cases: Optional[int] = None  # 限制本次评估的测试用例数量
    device_lookup_max_results: int = 3
    default_weather_location: str = "quebec city, Canada"


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
            "[Factual summary only; do not infer or speculate]",
            f"Command focus: {user_command}",
            f"Preference snapshot: {preference_summary}",
            f"Device environment: {device_summary}",
            f"History context: {history_summary}",
            f"Device lookup insights: {device_lookup_summary}",
        ]

        return "\n".join(summary_sections)


class DeviceLookupTool:
    """基于 SmartThings Planner + DocManager 的设备检索工具"""

    def __init__(
        self,
        *,
        docmanager_cache_path: Path,
        max_results: int = 3,
        planner_llm_config: Optional[LLMConfig] = None,
    ):
        self.doc_manager = DocManager.from_json(docmanager_cache_path)
        self.max_results = max_results
        self.device_state: Dict[str, Any] = {}
        self.docmanager_cache_path = docmanager_cache_path
        self.smartthings_planner = None
        if planner_llm_config is not None:
            self.smartthings_planner = self._build_smartthings_planner(
                planner_llm_config
            )

    def update_device_state(self, device_state: Optional[Dict[str, Any]]):
        self.device_state = device_state or {}

    def run(self, query: str) -> str:
        if not query:
            return "DeviceLookupTool: query is empty."
        planner_result = self._planner_lookup(query)
        if planner_result is not None:
            return planner_result
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

    def _build_smartthings_planner(self, planner_llm_config: LLMConfig):
        _ensure_tool_global_config(self.docmanager_cache_path)
        planner_config = SmartThingsPlannerToolConfig(
            llm_config=planner_llm_config,
        )
        try:
            return planner_config.instantiate()
        except Exception as exc:
            CONSOLE.log(f"[yellow]SmartThingsPlannerTool 初始化失败，退回关键词检索: {exc}")
            return None

    def _planner_lookup(self, query: str) -> Optional[str]:
        if self.smartthings_planner is None:
            return None
        try:
            plan_output = self.smartthings_planner.run(query.strip())
        except Exception as exc:
            CONSOLE.log(f"[yellow]SmartThings planner 查询失败: {exc}")
            return None

        device_ids = self._extract_device_ids_from_plan(plan_output)
        if not device_ids:
            return None

        lines = []
        for device_id in device_ids[: self.max_results]:
            metadata = self._build_device_metadata(device_id)
            lines.append(
                self._format_device_line(
                    device_id=device_id,
                    metadata=metadata,
                    score=5.0,  # Planner 已经筛选，无需再算分
                )
            )
        plan_details = self._extract_section(plan_output, "Plan")
        explanation = self._extract_section(plan_output, "Explanation")
        if plan_details:
            lines.append(f"Planner plan: {plan_details}")
        if explanation:
            lines.append(f"Planner notes: {explanation}")
        return "\n".join(lines)

    def _extract_device_ids_from_plan(self, plan_output: str) -> List[str]:
        device_section = self._extract_section(plan_output, "Device Ids")
        if not device_section:
            return []
        candidates = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", device_section, flags=re.IGNORECASE)
        seen = set()
        ordered = []
        for device_id in candidates:
            device_id = device_id.lower()
            if device_id in seen:
                continue
            if device_id not in self.doc_manager.device_names:
                continue
            seen.add(device_id)
            ordered.append(device_id)
        return ordered

    def _extract_section(self, text: str, header: str) -> Optional[str]:
        pattern = rf"{header}:(.*?)(?:\n[A-Z][A-Za-z ]+:|\Z)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        content = match.group(1).strip()
        return content if content else None


def _is_device_lookup_failure(message: str) -> bool:
    """快速检测设备检索是否失败（用于提示 planner 避免重复查询）。"""
    if not message:
        return True
    stripped = message.strip()
    lowered = stripped.lower()
    failure_markers = [
        "未找到",
        "not found",
        "unavailable",
        "query is empty",
        "error",
    ]
    if stripped.startswith("DeviceLookupTool:"):
        return True
    return any(marker in lowered for marker in failure_markers)


def _summarize_device_lookup_notes(
    device_lookup_notes: Optional[List[str]], max_entries: int = 3
) -> str:
    """将最近的设备检索结果压缩成提示文本，方便 planner 复用或调整。"""
    if not device_lookup_notes:
        return "(no device lookup attempts yet)"

    tail = device_lookup_notes[-max_entries:]
    start_idx = len(device_lookup_notes) - len(tail) + 1
    lines = []
    if len(device_lookup_notes) > max_entries:
        lines.append(
            f"(showing last {len(tail)} of {len(device_lookup_notes)} attempts)"
        )

    for offset, note in enumerate(tail, start=start_idx):
        label = "FAIL" if _is_device_lookup_failure(note) else "OK"
        stripped = note.strip()
        if not stripped:
            snippet = "(empty result)"
        else:
            first_line = stripped.splitlines()[0]
            snippet = first_line if len(first_line) < 160 else first_line[:157] + "..."
            if "\n" in stripped:
                snippet += " ..."
        lines.append(f"{label}#{offset}: {snippet}")

    return "\n".join(lines)


def _summarize_failure_notes(
    failure_notes: Optional[List[str]], max_entries: int = 3
) -> str:
    if not failure_notes:
        return "(no failed or exhausted retrievals)"

    tail = failure_notes[-max_entries:]
    lines: List[str] = []
    if len(failure_notes) > max_entries:
        lines.append(
            f"(showing last {len(tail)} of {len(failure_notes)} failures/exhausted attempts)"
        )

    start_idx = len(failure_notes) - len(tail) + 1
    for offset, note in enumerate(tail, start=start_idx):
        snippet = _shorten_text(note, 180) or "(empty failure note)"
        lines.append(f"#{offset}: {snippet}")
    return "\n".join(lines)


def _is_weather_lookup_failure(message: str) -> bool:
    if not message:
        return True
    lowered = message.lower()
    if lowered.startswith("weatherlookuptool error"):
        return True
    failure_markers = [
        "unavailable",
        "unable to find",
        "missing or invalid",
    ]
    return any(marker in lowered for marker in failure_markers)


class WeatherLookupTool:
    """封装 OpenWeatherMap 的简易天气检索工具"""

    def __init__(self, default_location: str = "quebec city, Canada"):
        self.default_location = default_location
        try:
            self.api = OpenWeatherMapAPIWrapper()
            self.available = True
        except Exception as exc:
            CONSOLE.log(f"[red]初始化 WeatherLookupTool 失败: {exc}")
            self.available = False
            self.api = None

    def run(self, location: Optional[str]) -> str:
        if not self.available or self.api is None:
            return "WeatherLookupTool unavailable: OpenWeatherMap API key missing or invalid."
        query = (location or self.default_location).strip()
        if not query:
            query = self.default_location
        try:
            report = self.api.run(query)
            return f"Weather report for {query}:\n{report}"
        except Exception as exc:
            return (
                f"WeatherLookupTool error for '{query}': {exc}. "
                "Ensure input follows 'City, Country'."
            )


class _InitialStateCapture(Exception):
    """内部异常，用于截获 testcase.setup 调用后的设备状态"""


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


def _extract_capability_value(components: Dict[str, Any], capability: str, attribute: str) -> Optional[Any]:
    """从设备组件信息中提取指定能力属性的最近值。"""
    for comp_data in components.values():
        cap_data = comp_data.get(capability)
        if not cap_data:
            continue
        attr_data = cap_data.get(attribute)
        if isinstance(attr_data, dict):
            return attr_data.get("value")
    return None


def _build_device_state_focus(
    device_state: Optional[Dict[str, Any]],
    doc_manager: Optional[DocManager],
    max_devices: int = 8,
) -> str:
    """
    构建强调 on/off、音量、亮度等关键信息的设备状态摘要，便于上游 prompt 明确当前局势。
    """
    if not device_state:
        return "(device state unavailable)"

    entries: List[Tuple[int, str]] = []
    for device_id, components in device_state.items():
        name = (
            doc_manager.device_names.get(device_id, device_id)
            if doc_manager and getattr(doc_manager, "device_names", None)
            else device_id
        )
        switch_val = _extract_capability_value(components, "switch", "switch")
        level_val = _extract_capability_value(components, "switchLevel", "level")
        volume_val = _extract_capability_value(components, "audioVolume", "volume")
        mute_val = _extract_capability_value(components, "audioMute", "mute")
        channel_val = _extract_capability_value(components, "tvChannel", "tvChannel")

        status_parts = []
        if switch_val is not None:
            status_parts.append(f"switch={switch_val}")
        if level_val is not None:
            status_parts.append(f"level={level_val}")
        if volume_val is not None:
            status_parts.append(f"volume={volume_val}")
        if mute_val is not None:
            status_parts.append(f"mute={mute_val}")
        if channel_val:
            status_parts.append(f"channel={channel_val}")

        if not status_parts:
            continue

        priority = 0
        if isinstance(switch_val, str) and switch_val.lower() == "on":
            priority += 3
        if isinstance(level_val, (int, float)) and level_val > 0:
            priority += 1
        if isinstance(volume_val, (int, float)) and volume_val > 0:
            priority += 1

        entries.append((priority, f"- {name} ({device_id}): {', '.join(status_parts)}"))

    if not entries:
        return "(no actionable device states found)"

    entries.sort(key=lambda item: item[0], reverse=True)
    trimmed = [line for _, line in entries[:max_devices]]
    if len(entries) > max_devices:
        trimmed.append(f"(其余 {len(entries) - max_devices} 台设备省略)")
    return "\n".join(trimmed)


def _serialize_for_compare(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _shorten_text(text: str, limit: int = 120) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _sanitize_summary_text(value: Any) -> Any:
    """
    Remove lines that look like reasoning, questions, or speculative offers from
    tool outputs that are expected to be factual summaries.
    """
    if not isinstance(value, str):
        return value
    lines = []
    speculative_markers = (
        "would you",
        "should i",
        "could you",
        "can i",
        "let me",
        "do you want",
        "shall i",
        "maybe",
        "perhaps",
        "seems like",
        "i can",
        "i will",
        "i suggest",
    )
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower_line = line.lower()
        if "?" in line:
            continue
        if any(marker in lower_line for marker in speculative_markers):
            continue
        lines.append(line)
    return "\n".join(lines)


MAX_REPEAT_QUERIES = 3
MAX_FAILURES_PER_ACTION = 2
REQUIRED_RETRIEVALS = ("preference_lookup", "device_lookup")
def _normalize_query_text(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return normalized


def _record_failure_note(
    action: str,
    message: str,
    failure_notes: List[str],
    action_failure_counts: Dict[str, int],
) -> None:
    failure_notes.append(message)
    action_failure_counts[action] = action_failure_counts.get(action, 0) + 1


def _should_skip_query(
    action: str,
    query: Optional[str],
    query_attempts: Dict[Tuple[str, str], int],
    failure_notes: List[str],
    action_failure_counts: Dict[str, int],
) -> bool:
    normalized = _normalize_query_text(query)
    if not normalized:
        return False
    key = (action, normalized)
    current = query_attempts.get(key, 0)
    if current >= MAX_REPEAT_QUERIES:
        _record_failure_note(
            action,
            f"{action} '{_shorten_text(query, 80)}' skipped: repeated {current} times without new info",
            failure_notes,
            action_failure_counts,
        )
        return True
    query_attempts[key] = current + 1
    return False


def _pick_missing_retrieval(
    action_counts: Dict[str, int],
    user_preferences: Any,
    device_facts: List[str],
) -> Optional[str]:
    if action_counts.get("preference_lookup", 0) == 0 and not user_preferences:
        return "preference_lookup"
    if action_counts.get("device_lookup", 0) == 0 and not device_facts:
        return "device_lookup"
    return None


DEVICE_CATEGORY_RULES: Dict[str, Dict[str, Any]] = {
    "lights": {
        "label": "灯光",
        "command_keywords": (
            "light",
            "lights",
            "lamp",
            "lighting",
            "chandelier",
            "sconce",
            "bedside light",
            "dining light",
        ),
        "name_keywords": (
            "light",
            "lamp",
            "sconce",
            "chandelier",
            "bulb",
        ),
        "capability_keywords": (
            "switchlevel",
            "colorcontrol",
            "colortemperature",
            "switch",
        ),
        "state_attributes": [
            ("switch", "switch", "switch"),
            ("switchLevel", "level", "level"),
            ("colorTemperature", "colorTemperature", "color_temp"),
            ("colorControl", "hue", "hue"),
            ("colorControl", "saturation", "saturation"),
        ],
        "attribute_keywords": {
            "switch": ("turn off", "turn on", "power", "switch"),
            "level": ("dim", "bright", "brightness", "level"),
            "color_temp": ("warm", "cool", "temperature", "white"),
        },
    },
    "tv": {
        "label": "电视",
        "command_keywords": (
            "tv",
            "television",
            "frame tv",
            "channel",
            "volume",
            "screen",
        ),
        "name_keywords": (
            "tv",
            "television",
            "frame",
            "screen",
        ),
        "capability_keywords": (
            "audiovolume",
            "audiomute",
            "mediainputsource",
            "tvchannel",
            "mediaplayback",
        ),
        "state_attributes": [
            ("switch", "switch", "switch"),
            ("audioVolume", "volume", "volume"),
            ("audioMute", "mute", "mute"),
            ("tvChannel", "tvChannel", "channel"),
            ("mediaInputSource", "inputSource", "input"),
        ],
        "attribute_keywords": {
            "volume": (
                "volume",
                "quieter",
                "louder",
                "loud",
                "soft",
            ),
            "channel": ("channel", "station"),
            "input": ("input", "source", "hdmi"),
            "switch": ("turn on", "turn off", "power"),
        },
    },
    "dishwasher": {
        "label": "洗碗机",
        "command_keywords": (
            "dishwasher",
            "dishes",
            "dish washing",
        ),
        "name_keywords": (
            "dishwasher",
            "dishes",
        ),
        "capability_keywords": (
            "dishwasheroperatingstate",
            "switch",
            "execute",
        ),
        "state_attributes": [
            ("switch", "switch", "switch"),
            ("dishwasherOperatingState", "machineState", "machine_state"),
            ("dishwasherOperatingState", "cycleRemainingTime", "remaining_time"),
        ],
        "attribute_keywords": {
            "machine_state": ("mode", "cycle", "phase", "state"),
            "switch": ("start", "stop", "turn on", "turn off"),
        },
    },
    "fridge": {
        "label": "冰箱",
        "command_keywords": (
            "fridge",
            "refrigerator",
            "freezer",
        ),
        "name_keywords": (
            "fridge",
            "refrigerator",
            "freezer",
        ),
        "capability_keywords": (
            "temperaturemeasurement",
            "thermostatcoolingsetpoint",
            "contactsensor",
        ),
        "state_attributes": [
            ("temperatureMeasurement", "temperature", "temperature"),
            ("thermostatCoolingSetpoint", "coolingSetpoint", "cooling_setpoint"),
            ("contactSensor", "contact", "door"),
        ],
        "attribute_keywords": {
            "temperature": ("temperature", "degree", "cold"),
            "cooling_setpoint": ("set", "adjust", "change"),
            "door": ("door", "open", "close"),
        },
    },
    "fireplace": {
        "label": "壁炉/火炉",
        "command_keywords": (
            "fireplace",
            "fire place",
            "fire",
        ),
        "name_keywords": (
            "fireplace",
            "fire place",
        ),
        "capability_keywords": (
            "switch",
            "switchlevel",
        ),
        "state_attributes": [
            ("switch", "switch", "switch"),
            ("switchLevel", "level", "level"),
        ],
        "attribute_keywords": {
            "switch": ("turn on", "turn off", "ignite", "power"),
            "level": ("dim", "bright", "intensity", "level"),
        },
    },
}


def _detect_target_device_categories(command: str) -> List[str]:
    if not command:
        return []
    lowered = command.lower()
    detected: List[str] = []
    for key, rule in DEVICE_CATEGORY_RULES.items():
        for keyword in rule.get("command_keywords", []):
            if keyword and keyword in lowered:
                detected.append(key)
                break
    return detected


def _infer_device_categories_from_metadata(
    device_name: str,
    doc_capabilities: Optional[List[Dict[str, Any]]],
    components: Optional[Dict[str, Any]],
) -> Set[str]:
    categories: Set[str] = set()
    name_lower = device_name.lower()
    capability_tokens: Set[str] = set()
    for cap in doc_capabilities or []:
        cap_id = cap.get("capability_id")
        if cap_id:
            capability_tokens.add(cap_id.lower())
    if components:
        for comp_data in components.values():
            for cap_name in comp_data.keys():
                capability_tokens.add(cap_name.lower())

    for key, rule in DEVICE_CATEGORY_RULES.items():
        if any(kw in name_lower for kw in rule.get("name_keywords", [])):
            categories.add(key)
            continue
        if capability_tokens.intersection(rule.get("capability_keywords", [])):
            categories.add(key)
    return categories


def _select_attribute_specs(
    rule: Dict[str, Any],
    command_lower: str,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    focus_labels: List[str] = []
    for label, keywords in (rule.get("attribute_keywords") or {}).items():
        for keyword in keywords:
            if keyword and keyword in command_lower:
                focus_labels.append(label)
                break
    specs = [
        spec for spec in rule.get("state_attributes", []) if spec[2] in focus_labels
    ]
    if not specs:
        specs = rule.get("state_attributes", [])
    return specs, focus_labels


def _format_state_values_for_specs(
    components: Dict[str, Any],
    specs: List[Tuple[str, str, str]],
) -> List[str]:
    values: List[str] = []
    for capability, attribute, label in specs:
        value = _extract_capability_value(components, capability, attribute)
        if value is None:
            continue
        values.append(f"{label}={value}")
    return values


def _collect_target_device_context(
    command: str,
    categories: List[str],
    device_state: Optional[Dict[str, Any]],
    doc_manager: Optional[DocManager],
) -> str:
    if not device_state:
        return "(device state unavailable)"
    if doc_manager is None:
        return "(DocManager unavailable for device context)"
    if not categories:
        return "(no device category keywords detected in command)"

    command_lower = command.lower()
    lines: List[str] = []

    for category_key in categories:
        rule = DEVICE_CATEGORY_RULES.get(category_key)
        if not rule:
            continue
        specs, focus_labels = _select_attribute_specs(rule, command_lower)
        if focus_labels:
            focus_text = ", ".join(dict.fromkeys(focus_labels))
        else:
            focus_text = ", ".join(
                dict.fromkeys(spec[2] for spec in specs)
            )
        lines.append(f"[{rule['label']}] 重点属性: {focus_text or '状态未知'}")
        matched = False
        for device_id, components in device_state.items():
            device_name = (
                doc_manager.device_names.get(device_id, device_id)
                if getattr(doc_manager, "device_names", None)
                else device_id
            )
            doc_caps = doc_manager.device_capabilities.get(device_id) if getattr(
                doc_manager, "device_capabilities", None
            ) else None
            device_categories = _infer_device_categories_from_metadata(
                device_name,
                doc_caps,
                components,
            )
            if category_key not in device_categories:
                continue
            values = _format_state_values_for_specs(components, specs)
            if not values:
                continue
            matched = True
            lines.append(f"- {device_name}: {', '.join(values)}")
        if not matched:
            lines.append("- (未找到匹配设备或缺少可用状态)")
    return "\n".join(lines)


def _build_target_device_context(
    command: str,
    device_state: Optional[Dict[str, Any]],
    doc_manager: Optional[DocManager],
) -> Tuple[List[str], str]:
    categories = _detect_target_device_categories(command)
    context = _collect_target_device_context(
        command=command,
        categories=categories,
        device_state=device_state,
        doc_manager=doc_manager,
    )
    return categories, context


def _ensure_tool_global_config(docmanager_cache_path: Path) -> None:
    """确保 smartthings 相关工具具备所需的 global_config 配置。"""
    if getattr(BaseConfig, "global_config", None) is None:
        BaseConfig.global_config = GlobalConfig()

    if (
        BaseConfig.global_config.docmanager_cache_path is None
        or BaseConfig.global_config.docmanager_cache_path != docmanager_cache_path
    ):
        BaseConfig.global_config.docmanager_cache_path = docmanager_cache_path

    if BaseConfig.global_config.condition_server_url is None:
        BaseConfig.global_config.condition_server_url = os.getenv(
            "CONDITION_SERVER_URL", "http://localhost:5001"
        )

    if BaseConfig.global_config.logpath is None:
        log_root = Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
            "logs", "rag_smartthings_planner"
        )
        log_root.mkdir(parents=True, exist_ok=True)
        BaseConfig.global_config.logpath = str(log_root)

@dataclass
class _LightweightTestConfig:
    """仿照 test_runner 中的 TestDemoConfig，仅保留评估所需的最小字段。"""

    coordinator_config: Any
    evaluator_llm: Any


_STATE_CAPTURE_TEST_CONFIG: Optional[_LightweightTestConfig] = None


def _get_state_capture_test_config() -> _LightweightTestConfig:
    """
    构造与 test_runner 行为一致的最小配置，确保 coordinator_config 可用。
    该配置会被缓存复用，避免重复构造开销。
    """

    global _STATE_CAPTURE_TEST_CONFIG
    if _STATE_CAPTURE_TEST_CONFIG is not None:
        return _STATE_CAPTURE_TEST_CONFIG

    if getattr(BaseConfig, "global_config", None) is None:
        BaseConfig.global_config = GlobalConfig()

    if BaseConfig.global_config.condition_server_url is None:
        BaseConfig.global_config.condition_server_url = os.getenv(
            "CONDITION_SERVER_URL", "http://localhost:5001"
        )

    if BaseConfig.global_config.logpath is None:
        BaseConfig.global_config.logpath = str(
            Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
                "logs", "rag_state_capture"
            )
        )

    coordinator_config = SAGECoordinatorConfig(
        llm_config=GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False),
        run_mode="test",
        enable_human_interaction=True,
        enable_google=False,
    )

    evaluator_stub = lambda *_args, **_kwargs: SimpleNamespace(
        content="stubbed evaluator (unused during state capture)"
    )

    _STATE_CAPTURE_TEST_CONFIG = _LightweightTestConfig(
        coordinator_config=coordinator_config,
        evaluator_llm=evaluator_stub,
    )
    return _STATE_CAPTURE_TEST_CONFIG


def _prepare_device_state_for_test(test_case: "TestCaseInfo") -> Dict[str, Any]:
    """
    尝试复用 testcases.py 中的初始化逻辑，生成特定测试用例的设备状态。
    我们在调用 testcase.setup 前截获 device_state，从而获得与真实测试一致的初始状态。
    """
    if getattr(test_case, "_prepared_state", None) is not None:
        return deepcopy(test_case._prepared_state)

    base_state = deepcopy(get_base_device_state())

    try:
        from sage.testing import testcases as testcase_module
    except Exception as exc:
        CONSOLE.log(f"[red]导入 testcases 失败，使用默认设备状态: {exc}")
        test_case._prepared_state = base_state
        return deepcopy(base_state)

    test_func = getattr(testcase_module, test_case.name, None)
    if not callable(test_func):
        test_case._prepared_state = base_state
        return deepcopy(base_state)

    original_setup = getattr(testcase_module, "setup", None)
    captured_state: Optional[Dict[str, Any]] = None

    def fake_setup(device_state, *_args, **_kwargs):
        nonlocal captured_state
        captured_state = deepcopy(device_state)
        raise _InitialStateCapture

    testcase_module.setup = fake_setup  # type: ignore[attr-defined]

    try:
        dummy_config = deepcopy(_get_state_capture_test_config())
        try:
            test_func(deepcopy(base_state), dummy_config)
        except _InitialStateCapture:
            pass
        except Exception as exc:
            CONSOLE.log(
                f"[red]执行 {test_case.name} 初始化逻辑失败，使用默认状态: {exc}"
            )
            captured_state = None
    finally:
        if original_setup is not None:
            testcase_module.setup = original_setup  # type: ignore[attr-defined]

    prepared = captured_state or base_state
    test_case._prepared_state = prepared
    return deepcopy(prepared)


def _load_tv_guide() -> List[Dict[str, str]]:
    global _TV_GUIDE_CACHE
    if _TV_GUIDE_CACHE is not None:
        return _TV_GUIDE_CACHE
    entries: List[Dict[str, str]] = []
    if not TV_GUIDE_PATH.exists():
        _TV_GUIDE_CACHE = entries
        return entries
    with TV_GUIDE_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    _TV_GUIDE_CACHE = entries
    return entries


def build_tv_guide_knowledge(user_command: str, max_entries: int = 10) -> str:
    """
    根据用户指令检索电视节目表，返回自然语言知识摘要。
    """
    entries = _load_tv_guide()
    if not entries:
        return "(tv guide unavailable)"

    keywords = {
        token.lower()
        for token in re.findall(r"[A-Za-z]+", user_command)
        if token
    }

    def score_entry(entry: Dict[str, str]) -> float:
        haystack = " ".join(
            [
                entry.get("channel_name", ""),
                entry.get("program_name", ""),
                entry.get("program_desc", ""),
            ]
        ).lower()
        if not haystack:
            return 0.0
        match_count = sum(1 for kw in keywords if kw and kw in haystack)
        return match_count + 0.1  # 平滑，保证在无匹配时保持原顺序

    ranked = sorted(entries, key=score_entry, reverse=True)
    selected = ranked[:max_entries] if keywords else entries[:max_entries]
    lines = []
    for entry in selected:
        lines.append(
            f"频道{entry.get('channel_number')} {entry.get('channel_name')} → "
            f"{entry.get('program_name')} | {entry.get('program_desc')}"
        )
    return "\n".join(lines)


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
    context_summary_text = (
        context_summary if context_summary else "(context summary unavailable)"
    )
    device_lookup_text = (
        "(device lookup tool not used)"
        if not device_lookup_notes
        else "\n".join(device_lookup_notes)
    )
    environment_text = (
        environment_overview if environment_overview else "(environment overview unavailable)"
    )
    tv_guide_text = (
        tv_guide_knowledge if tv_guide_knowledge else "(tv guide knowledge unavailable)"
    )
    weather_text = (
        "\n".join(weather_reports)
        if weather_reports
        else "(weather lookup not used)"
    )
    device_state_focus_text = (
        device_state_focus if device_state_focus else "(device state focus unavailable)"
    )
    target_device_context_text = (
        target_device_context if target_device_context else "(targeted device context unavailable)"
    )

    return f"""You are the intent analysis module. Base your reasoning ONLY on the facts below and eliminate any ambiguity.
History snippets are low-confidence hints; do not treat them as confirmation unless device facts match.
Call out unresolved references or risks explicitly.

Facts:
- Command: "{test_case.user_command}"
- Preferences / device facts: {preferences_text}
- History (low-confidence hints): {memory_text}
- Device lookup: {device_lookup_text}
- Context summary: {context_summary_text}
- Environment overview: {environment_text}
- Device state focus: {device_state_focus_text}
- Targeted device context: {target_device_context_text}
- Weather: {weather_text}
- TV guide: {tv_guide_text}

Respond with exactly three lines:
Intent: <precise description of the desired result, including target devices/actions/thresholds>
Signals: <key clues, personalization hints, situational triggers that justify the intent>
Risk: <remaining ambiguity or risk; write None only if absolutely clear>"""


def build_environment_overview_prompt(
    *,
    test_case: TestCaseInfo,
    device_lookup_notes: Optional[List[str]] = None,
    device_state: Optional[Dict[str, Any]] = None,
    target_device_context: Optional[str] = None,
) -> str:
    """
    生成设备/环境自然语言概览，帮助 LLM 理解当前可操作的设备与状态。
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
    return f"""Produce a concise environment briefing (objective, no speculation).
Use only the data provided:
- User command: "{test_case.user_command}"
- Device lookup notes: {lookup_text}
- Device snapshot: {device_state_text}
- Targeted device context: {target_context_text}

Write two short paragraphs:
1) Detected devices with inferred locations/capabilities.
2) Their current states and actionable controls (power, volume, input source, mode, etc.)."""


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
    intent_analysis_text = (
        intent_analysis
        if intent_analysis
        else "(intent analysis unavailable)"
    )
    environment_text = (
        environment_overview
        if environment_overview
        else "(environment overview unavailable)"
    )
    tv_guide_text = (
        tv_guide_knowledge
        if tv_guide_knowledge
        else "(tv guide knowledge unavailable)"
    )
    weather_text = (
        "\n".join(weather_reports)
        if weather_reports
        else "(weather lookup not used)"
    )
    device_state_focus_text = (
        device_state_focus if device_state_focus else "(device state focus unavailable)"
    )
    target_device_context_text = (
        target_device_context
        if target_device_context
        else "(targeted device context unavailable)"
    )

    base_prompt = f"""You are the decision module of a smart home assistant.
Only the information retrieved by earlier planner/tool calls is available to you now.
Do not assume that the full list of devices was preloaded; rely solely on the targeted
facts that have already been fetched.

Your task:
1. Using the already retrieved user preferences, history snippets, and device state, understand the current user command.
2. Apply deep reasoning to infer the user's underlying intent and situational context.
3. Decide whether you need to call the `human_interaction_tool` to ask the user clarifying questions.

Core Reasoning Principles:
Before deciding to ask for clarification, you MUST perform deep reasoning using these principles:

1. **Situational Context Analysis**: Analyze the broader context implied by the command.
   - Identify situational cues: communication events, activity transitions, environmental states, or task contexts.
   - Infer corrective or facilitative intent: When users mention problems or transitions, infer actions that address the issue or facilitate the transition.
   - Apply reverse logic: Complaints about conditions imply the user wants the opposite state.

2. **Common Sense Reasoning**: Apply everyday logic and social conventions.
   - Use domain knowledge about typical user behaviors and preferences in similar situations.
   - Infer task-appropriate settings based on the nature of the requested activity.
   - Apply social conventions and cultural norms relevant to the context.

3. **Default Action Inference**: When a specific value is missing but the intent is clear, infer a reasonable default action.
   - For directional adjustments: Apply moderate changes when direction is clear but magnitude is unspecified.
   - For task-appropriate modes: Select modes that match the task characteristics.
   - For subjective terms: Use industry-standard defaults or mid-range values.

4. **Contextual Disambiguation**: Use available context to resolve ambiguity.
   - Resolve pronouns and implicit references by matching to recently mentioned items, active devices, or contextually prominent elements.
   - Assume temporal continuity: References maintain continuity with recent interactions unless explicitly contradicted.
   - Leverage device lookup results when available.

Risk-Aware Decision Rule:
Use `human_interaction_tool` ONLY when there exists a genuine, unresolvable ambiguity that prevents action.
This means the ambiguity cannot be resolved through:
- Deep reasoning about situational context
- Common sense inference
- Contextual disambiguation using available information
- Application of reasonable defaults

Before finalizing a "Do not need" decision, confirm that:
- Every critical requirement (target entity, action parameters, personalization constraints) is either explicitly provided or backed by high-confidence inference.
- No equally plausible alternative interpretation remains unresolved.
- Executing without clarification will not risk incorrect or unsafe behavior.

When in doubt between autonomous action and clarification, prefer calling `human_interaction_tool`.

Your STRONG bias should be to solve the command autonomously using deep reasoning and common sense.
Only ask for clarification when there is a genuine, unresolvable ambiguity that prevents action.

Below is the information already retrieved for you (do NOT call additional tools; reason only with what is provided):

- Retrieved user preference / device facts (results of previous `preference_lookup` calls, may be empty):
{preferences_text}

- Retrieved historical interaction snippets (most relevant to the current command, may be empty):
{memory_text}

- Device lookup findings (from fake_requests snapshot + DocManager metadata):
{device_lookup_text}

- Context understanding summary (generated from targeted retrievals):
{context_summary_text}

- Deep intent hypothesis (generated by the dedicated intent analysis module):
{intent_analysis_text}

- Environment briefing (devices, locations, states summarized from DocManager metadata):
{environment_text}
- Device state focus (real-time on/off/levels to resolve ambiguity):
{device_state_focus_text}

- Command-targeted device context (grouped states for mentioned device types):
{target_device_context_text}

- TV program guide knowledge base:
{tv_guide_text}

- Weather lookup findings:
{weather_text}

Using the above context, analyze the user command with explicit Chain-of-Thought reasoning:

User command: "{test_case.user_command}"

Reasoning steps:
1. **Surface-level analysis**: Identify key verbs, nouns, adjectives, pronouns, and explicit parameters in the command.
2. **Deep intent inference**: Analyze the situational context and underlying user intent. What is the user really trying to achieve? What does the situation imply about the desired action?
3. **Information gap analysis**: Identify what specific information is missing (if any). Can this gap be filled by common sense inference, situational context, reasonable defaults, or retrieved preferences/history?
4. **Ambiguity resolution**: Determine if any remaining ambiguity is truly blocking or can be resolved through inference. Explicitly check whether multiple plausible interpretations or unverifiable assumptions remain.
5. **Decision**: Decide whether to call `human_interaction_tool`. Default to "Do not need" only when all critical requirements are supported by evidence or high-confidence inference; otherwise choose clarification.

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
    retrieval_progress = context_state.get("retrieval_progress") or {}

    def _status(key: str) -> str:
        return "done" if retrieval_progress.get(key) else "pending"

    coverage_status = "\n".join(
        [
            f"- preference_lookup: {_status('preference_lookup')}",
            f"- device_lookup: {_status('device_lookup')}",
            f"- context_summary: {_status('context_summary')}",
            f"- weather_lookup: {_status('weather_lookup')}",
        ]
    )

    planner_prompt = f"""You are orchestrating a Chain-of-Thought tool-using loop for the smart home assistant.
Nothing besides the user command is preloaded. When you need device-specific facts or user preferences,
issue focused retrieval instructions via the available tools instead of trying to enumerate the entire home.
Gather only what is necessary, then hand off to the final decision prompt that determines whether the
`human_interaction_tool` is required.

Coverage rule:
- Call each retrieval tool at least once if it is still marked as pending and relevant.
- Once `preference_lookup`, `device_lookup`, and `context_summary` have each run, avoid repeating them unless you have new evidence.

Reasoning hints:
- Start from common sense: if a single device satisfies the command, move forward instead of asking for clarification.
- Use failure notes to pivot: if a lookup already failed, try a different angle or move to `final_decision`.
- Keep the loop short; favor one decisive pass over repeated probing.

Current user command: "{test_case.user_command}"

Retrieved context so far:
- User preference / device facts: {preference_status}
- Device lookup insights: {device_fact_status}
  Latest samples:
{indented_device_facts}
- Context summary: {summary_status}
- Weather lookup insights: {weather_status}
- Device state focus: {device_state_focus_text}
- Retrieval coverage:
{coverage_status}
- Retrieval issues already encountered:
{failure_notes_summary}

Available tools (call at most one per step):
1. preference_lookup -> parameters: {{"query": "<english query string>"}}.
   Use for any missing user preference, historical insight, or device metadata.
   If you do not provide `query`, the system will reuse the default template:
   "{preference_query_template}".
2. device_lookup -> parameters: {{"query": "<english description of the target device/action>"}}.
   Surfaces relevant SmartThings device IDs, capabilities, and the latest fake_requests device state
   snapshot to help ground ambiguous references like "it" or "the fridge".
3. weather_lookup -> parameters: {{"query": "<City, Country>"}}.
   Call when the user command depends on current weather or outdoor conditions. If omitted,
   the system will use the default location configured for evaluation.
4. context_summary -> no parameters. Generates a consolidated summary using the latest
   user command, retrieved preferences/device facts, and history snippets.
5. final_decision -> no parameters. Use ONLY when you have enough information to
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
    weather_lookup_tool: Optional[WeatherLookupTool] = None,
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
    intent_analysis: Optional[str] = None
    environment_overview: Optional[str] = None
    tv_guide_knowledge: str = build_tv_guide_knowledge(test_case.user_command)
    weather_facts: List[str] = []
    device_state = _prepare_device_state_for_test(test_case)
    device_lookup_tool.update_device_state(device_state)
    doc_manager = getattr(device_lookup_tool, "doc_manager", None)
    device_state_focus = _build_device_state_focus(
        device_state, doc_manager
    )
    target_device_categories, target_device_context = _build_target_device_context(
        command=test_case.user_command,
        device_state=device_state,
        doc_manager=doc_manager,
    )
    failure_notes: List[str] = []
    query_attempts: Dict[Tuple[str, str], int] = {}
    action_failure_counts: Dict[str, int] = {}
    halted_actions: Set[str] = set()
    retrieval_progress = {
        "preference_lookup": False,
        "device_lookup": False,
        "context_summary": False,
        "weather_lookup": False,
    }
    new_info_since_last_plan = False

    chain_history: List[Dict[str, Any]] = []
    planner_steps_limit = 15
    final_response_text = ""
    final_reasoning = ""
    predicted_needs_tool = False

    CONSOLE.rule(f"[bold blue]测试用例: {test_case.name}")
    CONSOLE.log(f"[bold]用户指令[/bold]: {test_case.user_command}")
    device_state_summary = _summarize_device_state(device_state)
    CONSOLE.log(f"[bold]设备状态摘要[/bold]: {device_state_summary}")

    def _generate_environment_overview():
        nonlocal environment_overview
        if environment_overview:
            return environment_overview
        env_prompt = build_environment_overview_prompt(
            test_case=test_case,
            device_lookup_notes=device_facts if device_facts else None,
            device_state=device_state,
            target_device_context=target_device_context,
        )
        env_message = HumanMessage(content=env_prompt)
        env_response = llm([env_message])
        environment_overview = (
            env_response.content
            if hasattr(env_response, "content")
            else str(env_response)
        )
        CONSOLE.log(f"[green]环境概览[/green]: {environment_overview}")
        return environment_overview

    def _generate_intent_analysis():
        nonlocal intent_analysis
        if intent_analysis:
            return intent_analysis
        _generate_environment_overview()
        intent_prompt = build_intent_analysis_prompt(
            test_case=test_case,
            user_preferences=user_preferences,
            user_memory_snippets=user_memory_snippets,
            context_summary=context_summary,
            device_lookup_notes=device_facts,
            environment_overview=environment_overview,
            tv_guide_knowledge=tv_guide_knowledge,
            weather_reports=weather_facts,
            device_state_focus=device_state_focus,
            target_device_context=target_device_context,
        )
        intent_message = HumanMessage(content=intent_prompt)
        intent_response = llm([intent_message])
        intent_analysis = (
            intent_response.content
            if hasattr(intent_response, "content")
            else str(intent_response)
        )
        CONSOLE.log(f"[green]用户意图分析[/green]: {intent_analysis}")
        return intent_analysis

    required_keys = ("preference_lookup", "device_lookup", "context_summary")

    for step_idx in range(1, planner_steps_limit + 1):
        required_complete = all(retrieval_progress.get(key) for key in required_keys)
        if required_complete and not new_info_since_last_plan:
            planner_decision = {
                "thought": "All required retrievals completed; forcing final decision.",
                "action": "final_decision",
                "query": None,
            }
            planner_text = json.dumps(planner_decision, ensure_ascii=False)
        else:
            planner_prompt = build_chain_planner_prompt(
                test_case=test_case,
                chain_history=chain_history,
                preference_query_template=preference_query,
                context_state={
                    "user_preferences": user_preferences,
                    "context_summary": context_summary,
                    "device_facts": device_facts,
                    "weather_facts": weather_facts,
                    "device_state_focus": device_state_focus,
                    "failure_notes": failure_notes,
                    "retrieval_progress": retrieval_progress,
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
            new_info_since_last_plan = False
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

        if action in {"preference_lookup", "device_lookup", "weather_lookup"}:
            if action_failure_counts.get(action, 0) >= MAX_FAILURES_PER_ACTION:
                if action not in halted_actions:
                    failure_notes.append(
                        f"{action} skipped: exceeded failure limit ({MAX_FAILURES_PER_ACTION})"
                    )
                    halted_actions.add(action)
                continue

        if action == "preference_lookup":
            query = planner_decision["query"] or preference_query
            CONSOLE.log(f"[yellow]调用 UserProfileTool，query: {query}")
            prev_pref_snapshot = _serialize_for_compare(user_preferences)
            if _should_skip_query(
                action,
                query,
                query_attempts,
                failure_notes,
                action_failure_counts,
            ):
                continue
            try:
                tool_input = json.dumps(
                    {"query": query, "user_name": effective_user_name},
                    ensure_ascii=False,
                )
                user_preferences = user_profile_tool.run(tool_input)
                user_preferences = _sanitize_summary_text(user_preferences)
                CONSOLE.log(f"[green]User preferences 更新: {user_preferences}")
            except Exception as exc:
                user_preferences = "(failed to retrieve user preferences)"
                CONSOLE.log(
                    f"[red]UserProfileTool 调用失败: {exc}"
                )
                _record_failure_note(
                    action,
                    f"preference_lookup '{_shorten_text(query, 80)}' failed: {exc}",
                    failure_notes,
                    action_failure_counts,
                )
                continue
            new_pref_snapshot = _serialize_for_compare(user_preferences)
            if new_pref_snapshot == prev_pref_snapshot:
                _record_failure_note(
                    action,
                    f"preference_lookup '{_shorten_text(query, 80)}' yielded no new info",
                    failure_notes,
                    action_failure_counts,
                )
            else:
                new_info_since_last_plan = True
            retrieval_progress["preference_lookup"] = True
            continue

        if action == "device_lookup":
            query = planner_decision["query"] or test_case.user_command
            if _should_skip_query(
                action,
                query,
                query_attempts,
                failure_notes,
                action_failure_counts,
            ):
                continue
            lookup_result = device_lookup_tool.run(query)
            device_facts.append(lookup_result)
            CONSOLE.log(f"[green]设备检索结果[/green]: {lookup_result}")
            if _is_device_lookup_failure(lookup_result):
                _record_failure_note(
                    action,
                    f"device_lookup '{_shorten_text(query, 80)}' failed or empty",
                    failure_notes,
                    action_failure_counts,
                )
            else:
                new_info_since_last_plan = True
            retrieval_progress["device_lookup"] = True
            continue

        if action == "weather_lookup":
            if weather_lookup_tool is None:
                CONSOLE.log("[red]Weather tool 未配置，忽略该动作")
                weather_facts.append("Weather tool unavailable during evaluation.")
                retrieval_progress["weather_lookup"] = True
            else:
                query = planner_decision["query"] or weather_lookup_tool.default_location
                if _should_skip_query(
                    action,
                    query,
                    query_attempts,
                    failure_notes,
                    action_failure_counts,
                ):
                    continue
                report = weather_lookup_tool.run(query)
                weather_facts.append(report)
                CONSOLE.log(f"[green]天气检索结果[/green]: {report}")
                if _is_weather_lookup_failure(report):
                    _record_failure_note(
                        action,
                        f"weather_lookup '{_shorten_text(query, 80)}' returned no usable data",
                        failure_notes,
                        action_failure_counts,
                    )
                else:
                    new_info_since_last_plan = True
                retrieval_progress["weather_lookup"] = True
            continue

        if action == "context_summary":
            prev_context_summary = context_summary
            context_summary = context_tool.run(
                user_command=test_case.user_command,
                user_preferences=user_preferences,
                device_state=device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
            context_summary = _sanitize_summary_text(context_summary)
            CONSOLE.log(f"[green]上下文摘要更新: {context_summary}")
            retrieval_progress["context_summary"] = True
            if context_summary != prev_context_summary:
                new_info_since_last_plan = True
            continue

        # final_decision 或 fallback
        if context_summary is None:
            context_summary = context_tool.run(
                user_command=test_case.user_command,
                user_preferences=user_preferences,
                device_state=device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
            context_summary = _sanitize_summary_text(context_summary)
            CONSOLE.log(f"[green]最终决策前生成上下文摘要: {context_summary}")
            retrieval_progress["context_summary"] = True

        _generate_intent_analysis()
        final_prompt = build_cot_prompt(
            test_case=test_case,
            user_preferences=user_preferences,
            user_memory_snippets=user_memory_snippets,
            examples=None,  # 零样本模式，不提供示例
            context_summary=context_summary,
            device_lookup_notes=device_facts,
            intent_analysis=intent_analysis,
            environment_overview=environment_overview,
            tv_guide_knowledge=tv_guide_knowledge,
            weather_reports=weather_facts,
            device_state_focus=device_state_focus,
            target_device_context=target_device_context,
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
                device_state=device_state,
                user_memory_snippets=user_memory_snippets,
                device_lookup_notes=device_facts,
            )
            context_summary = _sanitize_summary_text(context_summary)
            retrieval_progress["context_summary"] = True
        _generate_intent_analysis()
        final_prompt = build_cot_prompt(
            test_case=test_case,
            user_preferences=user_preferences,
            user_memory_snippets=user_memory_snippets,
            examples=None,  # 零样本模式，不提供示例
            context_summary=context_summary,
            device_lookup_notes=device_facts,
            intent_analysis=intent_analysis,
            environment_overview=environment_overview,
            tv_guide_knowledge=tv_guide_knowledge,
            weather_reports=weather_facts,
            device_state_focus=device_state_focus,
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
        "intent_analysis": intent_analysis,
        "environment_overview": environment_overview,
        "tv_guide_knowledge": tv_guide_knowledge,
        "weather_lookup_notes": weather_facts,
        "device_state_focus": device_state_focus,
        "target_device_context": target_device_context,
        "target_device_categories": target_device_categories,
        "failure_notes": failure_notes,
        "retrieval_progress": retrieval_progress,
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
                "intent_analysis": intent_analysis,
                "environment_overview": environment_overview,
                "tv_guide_knowledge": tv_guide_knowledge,
                "weather_lookup_notes": weather_facts,
                "device_state_focus": device_state_focus,
                "target_device_context": target_device_context,
                "target_device_categories": target_device_categories,
                "failure_notes": failure_notes,
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
        planner_llm_config=deepcopy(config.llm_config),
    )
    weather_lookup_tool = WeatherLookupTool(
        default_location=config.default_weather_location
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
            weather_lookup_tool=weather_lookup_tool,
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


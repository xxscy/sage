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
# VLM 设备消歧工具（可选）
try:
    from sage.smartthings.device_disambiguation import VlmDeviceDetector
except Exception:  # pragma: no cover - 环境缺依赖时降级
    VlmDeviceDetector = None


@dataclass
class RAGCOTConfig:
    """RAG COT链的配置"""
    llm_config: LLMConfig = GPTConfig(
        model_name="gpt-4o-mini",
        temperature=0.0,
        streaming=False,
    )
    test_types_to_include: Optional[List[str]] = None  # None表示包含所有类型
    only_human_interaction: bool = False
    enable_type_filter: bool = False
    # 如果不指定，将尝试从用户指令前缀中自动解析（如 "Abhisek : xxx"）
    user_name: str = "default_user"
    # 每条指令用于检索用户偏好的查询模板（发送给 LLM，使用英文）
    preference_query_template: str = (
        "Based on this command, infer what preference-related query should be used "
        "for retrieval: {user_command}"
    )
    max_test_cases: Optional[int] = None  # 限制本次评估的测试用例数量
    device_lookup_max_results: int = 3
    default_weather_location: str = "quebec city, Canada"
    planner_max_repeat_queries: int = 3
    planner_max_failures_per_action: int = 2
    enable_environment_overview: bool = True
    enable_intent_analysis: bool = True
    focus_test_names: Optional[List[str]] = None
    repeat_runs_per_test: int = 1


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
            return json.dumps({"devices": [], "error": "Query is empty"}, ensure_ascii=False, indent=2)
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
            return json.dumps({"devices": [], "error": f"No devices found matching query: {query}"}, ensure_ascii=False, indent=2)

        matches.sort(key=lambda item: item[0], reverse=True)
        limited = matches[: self.max_results]

        # 如果有图片且候选>1，尝试用 VLM 消歧将最相关的设备提前
        if VlmDeviceDetector is not None:
            image_folder = (
                Path(os.getenv("SMARTHOME_ROOT", "."))
                .joinpath("sage/testing/assets/images")
            )
            if image_folder.exists() and len(limited) > 1:
                device_ids = [device_id for _, device_id, _ in limited]
                try:
                    winner, _ = _vlm_pick_device(
                        command=query, device_ids=device_ids, image_folder=image_folder
                    )
                    if winner in device_ids:
                        # 重新排序：VLM 命中者优先
                        limited.sort(key=lambda x: 0 if x[1] == winner else 1)
                        # 给命中者一个小的分数加成，便于输出提示
                        limited = [
                            (
                                score + (5 if dev_id == winner else 0),
                                dev_id,
                                metadata,
                            )
                            for score, dev_id, metadata in limited
                        ]
                except Exception as exc:
                    CONSOLE.log(f"[yellow]VLM 消歧在 device_lookup 中失败，已忽略: {exc}")
        # 只输出设备信息的 JSON 格式，不包含推理内容
        devices = []
        for score, device_id, metadata in limited:
            device_info = {
                "device_id": device_id,
                "name": metadata["name"],
                "capabilities": metadata["capabilities"],
                "components": metadata["components"],
                "state": metadata["state_summary"],
                "score": round(score, 2),
            }
            devices.append(device_info)
        
        result = {"devices": devices}
        extra = len(matches) - len(limited)
        if extra > 0:
            result["truncated_count"] = extra
        
        return json.dumps(result, ensure_ascii=False, indent=2)

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
        """精简设备状态摘要，只显示关键操作状态值"""
        device_state = self.device_state.get(device_id)
        if not device_state:
            return "(no state)"

        # 提取关键状态值（类似 _build_device_state_focus 的逻辑）
        key_values = []
        for component, cap_dict in list(device_state.items())[:component_limit]:
            comp_values = []
            
            # 只提取关键能力的状态值
            switch_val = self._extract_capability_value(cap_dict, "switch", "switch")
            level_val = self._extract_capability_value(cap_dict, "switchLevel", "level")
            volume_val = self._extract_capability_value(cap_dict, "audioVolume", "volume")
            mute_val = self._extract_capability_value(cap_dict, "audioMute", "mute")
            channel_val = self._extract_capability_value(cap_dict, "tvChannel", "tvChannel")
            temp_val = self._extract_capability_value(cap_dict, "temperatureMeasurement", "temperature")
            
            if switch_val is not None:
                comp_values.append(f"switch={switch_val}")
            if level_val is not None:
                comp_values.append(f"level={level_val}")
            if volume_val is not None:
                comp_values.append(f"volume={volume_val}")
            if mute_val is not None:
                comp_values.append(f"mute={mute_val}")
            if channel_val is not None:
                comp_values.append(f"channel={channel_val}")
            if temp_val is not None:
                comp_values.append(f"temp={temp_val}")
            
            # 如果没有关键值，显示第一个能力的前2个属性作为后备
            if not comp_values:
                for cap_idx, (capability, attributes) in enumerate(list(cap_dict.items())[:1]):
                    attr_summary = self._format_attribute_values(attributes, limit=2)
                    if attr_summary and attr_summary != "(empty capability)":
                        comp_values.append(f"{capability}: {attr_summary}")
                        break
            
            if comp_values:
                key_values.append(f"{component}({', '.join(comp_values)})")
        
        if not key_values:
            return "(no actionable state)"
        
        result = " | ".join(key_values)
        # 限制总长度，避免过长
        if len(result) > 200:
            result = result[:197] + "..."
        return result
    
    def _extract_capability_value(self, cap_dict: Dict[str, Any], capability: str, attribute: str) -> Optional[Any]:
        """从能力字典中提取指定属性的值"""
        if capability not in cap_dict:
            return None
        attrs = cap_dict[capability]
        if not isinstance(attrs, dict):
            return None
        if attribute not in attrs:
            return None
        attr_val = attrs[attribute]
        if isinstance(attr_val, dict) and "value" in attr_val:
            return attr_val["value"]
        return attr_val

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
        # 精简能力列表，只显示前3个关键能力
        key_capabilities = ["switch", "switchLevel", "tvChannel", "audioVolume", "audioMute", 
                           "colorControl", "temperatureMeasurement"]
        capabilities = metadata["capabilities"]
        # 优先显示关键能力
        priority_caps = [c for c in capabilities if any(kc in c.lower() for kc in key_capabilities)][:3]
        other_caps = [c for c in capabilities if c not in priority_caps][:2]
        shown_caps = priority_caps + other_caps
        capability_preview = ", ".join(shown_caps) if shown_caps else "(no capability data)"
        if len(capabilities) > len(shown_caps):
            capability_preview += f" (+{len(capabilities) - len(shown_caps)} more)"
        
        state_summary = metadata["state_summary"]
        # 精简格式：只显示关键信息
        return (
            f"[{score:.2f}] {metadata['name']} | "
            f"Caps: {capability_preview} | "
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

        # 只输出设备信息的 JSON 格式，不包含推理内容
        devices = []
        for device_id in device_ids[: self.max_results]:
            metadata = self._build_device_metadata(device_id)
            device_info = {
                "device_id": device_id,
                "name": metadata["name"],
                "capabilities": metadata["capabilities"],
                "components": metadata["components"],
                "state": metadata["state_summary"],
            }
            devices.append(device_info)
        
        return json.dumps({"devices": devices}, ensure_ascii=False, indent=2)

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
        # Always fall back to default location used in testcases
        query = self.default_location.strip() or "quebec city, Canada"
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
    *,
    max_repeat_queries: int,
) -> bool:
    normalized = _normalize_query_text(query)
    if not normalized:
        return False
    key = (action, normalized)
    current = query_attempts.get(key, 0)
    if current >= max_repeat_queries:
        _record_failure_note(
            action,
            f"{action} '{_shorten_text(query, 80)}' skipped: repeated {current} times without new info",
            failure_notes,
            action_failure_counts,
        )
        return True
    query_attempts[key] = current + 1
    return False


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
    vlm_device_id: Optional[str] = None,
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
            # 若 VLM 已命中且该设备不匹配，则跳过同类设备
            if vlm_device_id and device_id != vlm_device_id:
                device_name_for_cat = doc_manager.device_names.get(device_id, device_id) if getattr(doc_manager, "device_names", None) else device_id
                doc_caps_for_cat = doc_manager.device_capabilities.get(device_id) if getattr(
                    doc_manager, "device_capabilities", None
                ) else None
                dev_cats_for_cat = _infer_device_categories_from_metadata(
                    device_name_for_cat,
                    doc_caps_for_cat,
                    components,
                )
                if category_key in dev_cats_for_cat:
                    continue
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
    vlm_device_id: Optional[str] = None,
) -> Tuple[List[str], str]:
    categories = _detect_target_device_categories(command)
    context = _collect_target_device_context(
        command=command,
        categories=categories,
        device_state=device_state,
        doc_manager=doc_manager,
        vlm_device_id=vlm_device_id,
    )
    if vlm_device_id and doc_manager and getattr(doc_manager, "device_names", None):
        device_name = doc_manager.device_names.get(vlm_device_id, vlm_device_id)
        context = f"[VLM] 视觉消歧命中: {device_name} ({vlm_device_id})\n" + context
    elif vlm_device_id:
        context = f"[VLM] 视觉消歧命中: {vlm_device_id}\n" + context
    return categories, context


def _has_device_image(device_id: str, image_folder: Path) -> bool:
    """
    检查设备是否有对应的图片文件（支持多种格式）。
    参考 device_disambiguation.py 的设计，使用文件名（不含扩展名）作为 device_id。
    """
    if not image_folder or not image_folder.exists():
        return False
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for ext in image_extensions:
        if image_folder.joinpath(f"{device_id}{ext}").exists():
            return True
    return False


def _get_available_image_device_ids(image_folder: Path) -> Set[str]:
    """
    获取图片文件夹中所有可用的设备ID（文件名不含扩展名）。
    参考 device_disambiguation.py 的 get_images 方法。
    """
    available_ids = set()
    if not image_folder or not image_folder.exists():
        return available_ids
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for file in image_folder.iterdir():
        if not file.is_file():
            continue
        if file.suffix in image_extensions:
            # 使用文件名（不含扩展名）作为 device_id
            device_id_from_file = file.stem
            available_ids.add(device_id_from_file)
    return available_ids


def _vlm_pick_device(
    *,
    command: str,
    device_ids: List[str],
    image_folder: Optional[Path],
) -> Tuple[Optional[str], Optional[str]]:
    """
    使用 VLM (CLIP) 对候选设备进行一次视觉消歧，返回 (winner_device_id, note)。

    - 仅在 VlmDeviceDetector 可用且 image_folder 存在时启用。
    - 若仅 0/1 个设备或图片缺失，返回 (None, None)。
    
    参考 device_disambiguation.py 的设计：
    1. 先创建 VlmDeviceDetector 实例
    2. 使用 get_images() 获取所有可用的图片文件名（作为 real_devices）
    3. 使用 select_devices() 方法匹配 device_ids 和图片文件名
    4. 只传入能够匹配到的设备 ID 给 identify_device_with_scores
    """
    if VlmDeviceDetector is None:
        return None, "[VLM error: detector unavailable]"
    if not image_folder or not image_folder.exists():
        return None, "[VLM error: image folder missing]"
    
    if len(device_ids) <= 1:
        return None, f"[VLM none: insufficient candidates (found {len(device_ids)} device(s))]"
    
    try:
        # 创建 VlmDeviceDetector 实例
        detector = VlmDeviceDetector(str(image_folder))
        
        # 获取所有可用的图片文件名（参考 device_disambiguation.py 的 get_images 方法）
        image_dict = detector.get_images()
        real_devices = list(image_dict.keys())
        
        if not real_devices:
            return None, "[VLM error: no images found in folder]"
        
        # 使用 select_devices 方法匹配 device_ids 和图片文件名
        # 这个方法会返回匹配到的图片文件名（real_devices 中的 ID）
        matched_device_list = detector.select_devices(device_ids, real_devices)
        
        if not matched_device_list:
            return None, f"[VLM error: none of the {len(device_ids)} device(s) matched any images. Available images: {len(real_devices)}"
        
        if len(matched_device_list) <= 1:
            # 只有一个匹配，直接返回
            winner = matched_device_list[0]
            return winner, f"[VLM single match: {winner} for '{command}']"
        
        # 构建 payload，传入匹配到的设备 ID（这些是图片文件名）
        payload = {
            "devices": matched_device_list,
            "disambiguation_information": command,
        }
        
        winner = None
        ranked: List[Tuple[str, float]] = []
        if hasattr(detector, "identify_device_with_scores"):
            winner, ranked = detector.identify_device_with_scores(json.dumps(payload))
        else:
            winner = detector.identify_device(json.dumps(payload))
            # 如果没有 ranked，创建一个默认的 ranked 列表
            if winner and isinstance(winner, str) and winner not in ["", "None"]:
                ranked = [(winner, 1.0)]
        
        # 检查 winner 是否是错误消息（字符串且不在 matched_device_list 中）
        if isinstance(winner, str) and winner not in matched_device_list:
            # winner 是错误消息，不是有效的 device_id
            error_msg = winner
            if ranked:
                score_log = ", ".join(f"{dev}:{score:.3f}" for dev, score in ranked[:5])
                return None, f"[VLM error: {error_msg}, scores={score_log}]"
            return None, f"[VLM error: {error_msg}]"
        
        if ranked:
            score_log = ", ".join(f"{dev}:{score:.3f}" for dev, score in ranked[:5])
            top_score = ranked[0][1]
            second_score = ranked[1][1] if len(ranked) > 1 else None
            score_gap = top_score - (second_score if second_score is not None else 0.0)
            CONSOLE.log(f"[cyan]VLM scores[/cyan]: {score_log}")
            # 阈值：分差<0.05 或 顶分<0.25 视为视觉歧义，保留前3供下游处理
            if (second_score is not None and score_gap < 0.05) or top_score < 0.25:
                top_candidates = [dev_id for dev_id, _ in ranked[:3]]
                return None, f"VLM ambiguous: top gap {score_gap:.3f}, top score {top_score:.3f}, candidates={top_candidates}, scores={score_log}"
        
        # 验证 winner 是否在匹配列表中
        if winner and winner in matched_device_list:
            return winner, f"VLM disambiguation suggests device {winner} for '{command}'. scores={', '.join(f'{d}:{s:.3f}' for d, s in ranked[:5])}" if ranked else f"VLM disambiguation suggests device {winner} for '{command}'."
    except Exception as exc:
        err = f"[VLM error: {exc}]"
        CONSOLE.log(f"[yellow]VLM 消歧失败，忽略: {exc}")
        import traceback
        CONSOLE.log(f"[yellow]详细错误信息: {traceback.format_exc()}")
        return None, err
    # ranked 为空或未返回 winner
    return None, "[VLM none: no match]" if not ranked else f"[VLM none: no match, scores={', '.join(f'{d}:{s:.3f}' for d, s in ranked[:5])}]"


def _vlm_disambiguate_devices(
    *,
    command: str,
    device_ids: List[str],
    image_folder: Optional[Path],
) -> Optional[str]:
    # 兼容旧调用：只返回 note
    _, note = _vlm_pick_device(command=command, device_ids=device_ids, image_folder=image_folder)
    return note


def _collect_image_hints(
    *,
    command: str,
    device_ids: List[str],
    doc_manager: Optional[DocManager],
    image_folder: Optional[Path],
    max_images: int = 3,
) -> List[str]:
    """
    根据文件名(即 device_id) 收集可用的设备图片，并用设备名/命令相似度做简单排序。
    仅提供“提示”文本（device_id -> image path），不做真正的视觉识别。
    """
    if not image_folder or not image_folder.exists():
        return []
    # 仅保留既有 device_id 又有图片的
    available_ids = set(device_ids)
    hints: List[tuple[float, str, str]] = []
    for file in image_folder.iterdir():
        if not file.is_file():
            continue
        stem = file.stem
        if stem not in available_ids:
            continue
        name = (
            doc_manager.device_names.get(stem, stem)
            if doc_manager and getattr(doc_manager, "device_names", None)
            else stem
        )
        # 简单相似度：命令与设备名
        score = SequenceMatcher(None, command.lower(), name.lower()).ratio()
        hints.append((score, stem, str(file)))
    hints.sort(key=lambda x: x[0], reverse=True)
    top = hints[:max_images]
    return [f"{dev_id} ({score:.2f}) -> {path}" for score, dev_id, path in top]


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
            f"Channel {entry.get('channel_number')} {entry.get('channel_name')} -> "
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

**Preference Handling Rules (CRITICAL):**
- **Explicit preference references**: If the user explicitly references a personal preference (e.g., "my favorite X", "my usual Y", "my preferred Z"), you MUST verify this preference is retrievable from the provided preferences/history.
  - If the preference is **found and verified** in the provided data -> Use it in the Refined Command.
  - If the preference is **not found or not retrievable** -> Mark as `UNKNOWN_PARAMETER` in the Refined Command. **Do NOT infer or guess** the preference from unrelated context or general knowledge.
  - **Rationale**: Personal preferences are user-specific and cannot be safely inferred; guessing wrong preferences violates user trust.

**Anti-Hallucination Guard (CRITICAL):**
- Ground every decision **only** on the provided facts (preferences, history, device lookup, device state, guide/knowledge). If a required fact is absent or ambiguous, mark it as `UNKNOWN_PARAMETER` rather than inventing it.
- Preference usage must be **explicitly supported** by the provided data for the same user and relevant device/context. Generic statements or weak associations do **not** authorize assuming a preference.
- If multiple candidate targets or preferences exist without a clear single winner, treat the target/preference as **ambiguous** and ask for clarification.
- When command uses singular references and multiple devices fit, you must treat the target as unresolved until unambiguous grounding is available.

**Execution Bias Guidelines (apply when not conflicting above):**
- Align actions with the user's expressed comfort/goal; adjust intensity downward when user indicates overload, upward when user indicates insufficiency.
- When direction is known but parameters are missing, apply conservative defaults; mark UNKNOWN when:
  - The user asked for a specific-but-unknown preference (highest priority)
  - No safe default can be inferred from context, device capabilities, or general patterns
- If VLM/lookup or naming/location yields a high-confidence single target, use it; similar devices do not block execution.
- When preferences/history specify an entity or mode but omit an attribute, infer the typical attribute and label it as inferred; leave unknown only when no reasonable basis exists.

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

**Critical Context Extraction Rules:**
- **Current state inference**: When the command references contextual state values (e.g., "this channel", "current channel", "the channel", or similar references to existing device state), you MUST extract the current value from Device Lookup Results or Device State Focus.
  - Look for state attributes that represent the current operational value (channel numbers, current levels, active inputs, etc.) in device state information.
  - If a device has a current operational value available in state, use that value in the Refined Command.
  - Only mark as UNKNOWN_PARAMETER if no current state information is available in the provided device state.
- **Contextual device references**: When the command uses contextual device references (e.g., "this device", "other device", "the device"), extract device identification from Device Lookup Results or Device State Focus.
  - Use device names, locations, active states, or operational status to identify contextual device references.
  - If context clearly identifies devices based on their operational state or contextual position, proceed with the identified devices.

**Output Structure:**
- **Request Type**: [Device Control / Content Consumption]
- **Identified Situation**: [Abstract description of user context]
- **Content Verification**: [Found verifiable match in Guide/History OR "No verifiable match found"]
- **Refined Command**: [Explicit command. Use 'UNKNOWN_PARAMETER' if verification fails.]
- **Confidence**: [High (Verified/Obvious) / Low (Unverified Content/Ambiguous Target)]
- **Reasoning**: [Step-by-step deduction. Explicitly state if TV Guide was used or if a default was applied.]
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

    base_prompt = f"""You are the autonomous decision module of a smart home assistant.
Your GOAL: Complete the user's objective. Your primary directive is to execute user commands, not to add functionality or ask unnecessary questions.

**Core Principle:**
- **Execute user intent**: Your purpose is to fulfill the user's goal, not to seek clarification unless absolutely necessary.
- **No feature additions**: Do not add functionality beyond what the user requested.
- **Prefer action over inquiry**: When a reasonable default can be inferred from context, device capabilities, or user intent, apply it and proceed. Only ask when the command is genuinely ambiguous and no safe default exists.

**Decision Protocol (Zero-Shot Logic):**
You must output `Do not need to use human_interaction_tool` if the `Refined Command` can be **reasonably completed** with available information, even if some parameters are inferred from context or use conservative defaults.

**Logic 0: Information Request Exemption** (Highest Priority):
- **Rule**: If the command's primary intent is to **retrieve or report information** about device state, status, or current values (rather than change state), it is complete as-is.
- **Semantic Test**: Does the command seek to know/read/check/obtain information without requiring state modification? If yes -> **Do not need**.
- **Rationale**: Information requests don't require parameters; they query existing state.

**Logic 1: The Content Verification Gate**:
- **Condition**: The request is classified as **Content Consumption** (watching/listening).
- **Check**: Review the `Deep Intent Hypothesis`. Did it successfully find a specific, verified source (channel/app) in the Knowledge Base?
- **Decision**:
  - If `Refined Command` contains a specific, verified source -> **Do not need** (Proceed).
  - If `Refined Command` contains "UNKNOWN" or relies on unverified assumptions -> **Need** (Ask clarification).
  - *Rationale*: Guessing the wrong content is a failure.

**Logic 2: The Enhanced Safe Default Exemption**:
- **Condition**: The request is classified as **Device Control** and a parameter is missing or ambiguous.
- **Decision Rules** (apply in order):
  1. **Singular target requirement**: When the command implies a single target, proceed only if one device is unambiguously grounded. If multiple devices fit and none is uniquely grounded, you MUST ask -> **Need**.
  2. **Relative value expressions**: If the command specifies a relative change and device state provides current values, compute from state -> **Do not need**. If state unavailable, apply reasonable absolute defaults -> **Do not need**.
  3. **Qualitative intent descriptors**: If the command uses subjective terms expressing user comfort or intent, infer context-appropriate defaults from user intent and device capabilities -> **Do not need**.
  4. **Temporal/conditional rule definitions**: If the command establishes a condition-action relationship, it defines a rule rather than immediate action -> **Do not need**.
  5. **Collective references**: If the command clearly targets a group (all/every/both) and context identifies that group, proceed -> **Do not need**.
  6. **Directional adjustments without magnitude**: If the command specifies direction but not exact value, apply conservative defaults aligned with the stated goal -> **Do not need**.
  7. **Explicit unknown preference requests**: If user explicitly references a specific-but-unknown personal preference and it is not retrievable -> **Need**.
  8. **Ambiguity with no safe default**: If the command remains ambiguous and no reasonable default can be inferred -> **Need**.

**Execution Bias (CRITICAL - apply when not conflicting with above):**
- **Primary directive: Complete user goals accurately**: Execute the user's command as intended, without inventing extra targets or preferences.
- **Respect singular targeting**: If a command implies one target and multiple candidates exist, do not act until a single target is grounded; ask if needed.
- **No target multiplication**: Do not expand a singular request into multiple actions across devices.
- **Context-aware defaults**: Align action direction with user-stated comfort/goal; reduce intensity when user indicates overload, increase when indicating insufficiency. Infer parameters only when the target is unambiguous.
- **Device grounding confidence**: Proceed only when a single target is clearly grounded (by name/location/state/VLM or uniqueness). If multiple candidates remain, ask.
- **Conservative parameter inference**: When direction is known but numeric parameters are missing, apply conservative defaults based on context, device capabilities, or general patterns. Mark UNKNOWN when a required parameter (especially personal preference) is not verifiably available.
- **Minimize interruptions, prevent wrong actions**: Prefer not to interrupt, but avoid executing incorrect or multi-target actions when the request is singular and ambiguous.

**Input Data:**
- User Command: "{test_case.user_command}"
- Deep Intent Hypothesis:
{intent_analysis_text}
- Device Context:
{device_state_focus_text}

**Reasoning Process:**
1. **Check Logic 0**: Determine if this is an information request -> If yes, **Do not need** (skip to conclusion).
2. **Use Intent Analysis Results**: The `Deep Intent Hypothesis` contains the `Refined Command` which has already resolved device ambiguity. **You MUST use the device(s) specified in the `Refined Command`** - do NOT re-analyze device ambiguity. If the `Refined Command` specifies a device, treat it as resolved.
3. **Classify Intent**: Is this Device Control or Content Consumption?
4. **Analyze Refinement**: Look at the `Refined Command` and `Content Verification` status from the `Deep Intent Hypothesis`. If a device is specified in `Refined Command`, it is already resolved - do not ask for device clarification.
5. **Apply Logic**: Use Logic 1 for Content, Logic 2 for Device Control. When applying Logic 2, if the `Refined Command` already specifies a device, treat device ambiguity as resolved.
6. **Final Verdict**: Determine if clarification is strictly required. Remember: if `Refined Command` specifies a device, device ambiguity is already resolved.

Format your answer exactly as:
Reasoning: Step 1 - Check Information Request: [...] Step 2 - Use Intent Analysis Results: [State which device(s) are specified in Refined Command, confirm device ambiguity is resolved] Step 3 - Classify Intent: [...] Step 4 - Analyze Refinement: [...] Step 5 - Apply Logic: [...] Step 6 - Final Verdict: [...]

Conclusion:
Need / Do not need to use human_interaction_tool

Reason:
[Brief explanation focusing on information request status, verification status, or safe defaults applied]

IMPORTANT: The line after `Conclusion:` MUST be exactly either `Need to use human_interaction_tool` or `Do not need to use human_interaction_tool`.
"""

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
    planner_max_repeat_queries: int = 3,
    planner_max_failures_per_action: int = 2,
    enable_environment_overview: bool = True,
    enable_intent_analysis: bool = True,
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
    context_summary: Optional[str] = "(context summary disabled)"
    intent_analysis: Optional[str] = None
    environment_overview: Optional[str] = "(environment overview disabled)"
    tv_guide_knowledge: str = build_tv_guide_knowledge(test_case.user_command)
    weather_facts: List[str] = []
    device_state = _prepare_device_state_for_test(test_case)
    device_state_for_prompt = device_state
    device_lookup_tool.update_device_state(device_state)
    doc_manager = getattr(device_lookup_tool, "doc_manager", None)
    device_state_focus = "(device state focus disabled)"
    target_device_categories, target_device_context = ([], "(target device context disabled)")
    failure_notes: List[str] = []
    query_attempts: Dict[Tuple[str, str], int] = {}
    action_failure_counts: Dict[str, int] = {}
    halted_actions: Set[str] = set()

    chain_history: List[Dict[str, Any]] = []
    planner_steps_limit = 15
    final_response_text = ""
    final_reasoning = ""
    predicted_needs_tool = False

    CONSOLE.rule(f"[bold blue]测试用例: {test_case.name}")
    CONSOLE.log(f"[bold]用户指令[/bold]: {test_case.user_command}")
    # Skip detailed device state summary to reduce overhead
    device_state_summary = "(device state summary disabled)"

    def _generate_environment_overview(
        vlm_hint: Optional[str] = None, image_hints: Optional[List[str]] = None
    ):
        # Disabled to reduce overhead
        return environment_overview

    def _generate_intent_analysis():
        nonlocal intent_analysis, environment_overview
        if intent_analysis:
            return intent_analysis
        if not enable_intent_analysis:
            intent_analysis = "(intent analysis disabled)"
            return intent_analysis
        _generate_environment_overview(vlm_hint, image_hints)
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

    # 可选：基于图片的 VLM 消歧提示
    image_folder = Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
        "sage/testing/assets/images"
    )
    vlm_hint: Optional[str] = None
    vlm_applied = False
    image_hints: List[str] = []

    def _collect_vlm_candidates(
        *, categories: List[str], device_state: Dict[str, Any], doc_manager: Optional[DocManager]
    ) -> List[str]:
        """仅收集与目标类别相关且有对应图片的设备 ID，用于 VLM 消歧。"""
        if not image_folder.exists():
            return []
        candidates: List[str] = []
        for dev_id, components in device_state.items():
            if not _has_device_image(dev_id, image_folder):
                continue
            name = (
                doc_manager.device_names.get(dev_id, dev_id)
                if doc_manager and getattr(doc_manager, "device_names", None)
                else dev_id
            )
            doc_caps = (
                doc_manager.device_capabilities.get(dev_id)
                if doc_manager and getattr(doc_manager, "device_capabilities", None)
                else None
            )
            dev_cats = _infer_device_categories_from_metadata(
                name,
                doc_caps,
                components,
            )
            if categories and not dev_cats.intersection(categories):
                continue
            candidates.append(dev_id)
        return candidates

    def _apply_vlm_after_lookups():
        """在积累完 device_facts 后再触发一次 VLM 消歧。"""
        nonlocal vlm_applied, vlm_hint, device_state_for_prompt, device_state_focus, image_hints
        nonlocal target_device_context, target_device_categories
        if vlm_applied:
            return
        vlm_applied = True
        initial_target_categories = _detect_target_device_categories(test_case.user_command)
        vlm_candidates = _collect_vlm_candidates(
            categories=initial_target_categories,
            device_state=device_state,
            doc_manager=doc_manager,
        )
        if len(vlm_candidates) < 2:
            vlm_candidates = [
                dev_id
                for dev_id in device_state.keys()
                if _has_device_image(dev_id, image_folder)
            ]

        vlm_winner, vlm_hint_local = (None, None)
        if len(vlm_candidates) >= 2:
            vlm_winner, vlm_hint_local = _vlm_pick_device(
                command=test_case.user_command,
                device_ids=vlm_candidates,
                image_folder=image_folder,
            )
        elif len(vlm_candidates) == 1:
            vlm_hint_local = "[VLM skipped: only one image candidate]"
        else:
            vlm_hint_local = "[VLM skipped: no image candidates]"

        vlm_hint = vlm_hint_local
        if vlm_hint:
            device_facts.append(f"[VLM] {vlm_hint}")
        image_hints = _collect_image_hints(
            command=test_case.user_command,
            device_ids=list(device_state.keys()),
            doc_manager=doc_manager,
            image_folder=image_folder,
        )

        if vlm_winner:
            filtered: Dict[str, Any] = {}
            for dev_id, comp in device_state.items():
                name = (
                    doc_manager.device_names.get(dev_id, dev_id)
                    if doc_manager and getattr(doc_manager, "device_names", None)
                    else dev_id
                )
                doc_caps = (
                    doc_manager.device_capabilities.get(dev_id)
                    if doc_manager and getattr(doc_manager, "device_capabilities", None)
                    else None
                )
                cats = _infer_device_categories_from_metadata(
                    name,
                    doc_caps,
                    comp,
                )
                if cats.intersection(_detect_target_device_categories(test_case.user_command)) and dev_id != vlm_winner:
                    continue
                filtered[dev_id] = comp
            if filtered:
                device_state_for_prompt = filtered

        # Skip detailed device state focus and target context updates

    # 初始设备摘要（在可能的 VLM 过滤前，已禁用细粒度摘要）
    device_state_focus = device_state_focus
    target_device_categories, target_device_context = target_device_categories, target_device_context

    for step_idx in range(1, planner_steps_limit + 1):
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
        if action in halted_actions:
            planner_decision["thought"] = (
                planner_decision.get("thought", "")
                + " | action halted due to no new info; forcing final_decision"
            ).strip()
            action = "final_decision"
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
            if action_failure_counts.get(action, 0) >= planner_max_failures_per_action:
                if action not in halted_actions:
                    failure_notes.append(
                        f"{action} skipped: exceeded failure limit ({planner_max_failures_per_action})"
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
                max_repeat_queries=planner_max_repeat_queries,
            ):
                halted_actions.add(action)
                continue
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
                _record_failure_note(
                    action,
                    f"preference_lookup '{_shorten_text(query, 80)}' failed: {exc}",
                    failure_notes,
                    action_failure_counts,
                )
                halted_actions.add(action)
                continue
            new_pref_snapshot = _serialize_for_compare(user_preferences)
            if new_pref_snapshot == prev_pref_snapshot:
                _record_failure_note(
                    action,
                    f"preference_lookup '{_shorten_text(query, 80)}' yielded no new info",
                    failure_notes,
                    action_failure_counts,
                )
                halted_actions.add(action)
            continue

        if action == "device_lookup":
            query = planner_decision["query"] or test_case.user_command
            if _should_skip_query(
                action,
                query,
                query_attempts,
                failure_notes,
                action_failure_counts,
                max_repeat_queries=planner_max_repeat_queries,
            ):
                halted_actions.add(action)
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
                halted_actions.add(action)
            continue

        if action == "weather_lookup":
            if weather_lookup_tool is None:
                CONSOLE.log("[red]Weather tool 未配置，忽略该动作")
                weather_facts.append("Weather tool unavailable during evaluation.")
            else:
                query = planner_decision["query"] or weather_lookup_tool.default_location
                if _should_skip_query(
                    action,
                    query,
                    query_attempts,
                    failure_notes,
                    action_failure_counts,
                    max_repeat_queries=planner_max_repeat_queries,
                ):
                    halted_actions.add(action)
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
                    halted_actions.add(action)
            continue

        if action == "context_summary":
            # Context summary disabled
            context_summary = "(context summary disabled)"
            continue

        # final_decision 或 fallback
        if context_summary is None:
            context_summary = "(context summary disabled)"

        _apply_vlm_after_lookups()
        _generate_intent_analysis()
        if environment_overview is None:
            environment_overview = "(environment overview disabled)"
        if intent_analysis is None and not enable_intent_analysis:
            intent_analysis = "(intent analysis disabled)"
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
            intent_reasoning_enabled=enable_intent_analysis,
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
        _apply_vlm_after_lookups()
        _generate_intent_analysis()
        if environment_overview is None:
            environment_overview = "(environment overview disabled)"
        if intent_analysis is None and not enable_intent_analysis:
            intent_analysis = "(intent analysis disabled)"
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
            intent_reasoning_enabled=enable_intent_analysis,
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
        f"{'需要' if predicted_needs_tool else '不需要'} "        f"(期望: {'需要' if test_case.requires_human_interaction else '不需要'})"
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
        "tv_guide_knowledge": tv_guide_knowledge,
        "weather_lookup_notes": weather_facts,
        "failure_notes": failure_notes,
        "vlm_notes": vlm_hint,
    }

    # 将日志写入文件（按测试用例名分类）
    if log_base_dir is not None:
        try:
            case_folder = _sanitize_filename(test_case.name)
            case_dir = log_base_dir.joinpath(case_folder)
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
                "reasoning": final_reasoning,
                "llm_response": final_response_text,
                "chain_history": chain_history,
                "device_lookup_notes": device_facts,
                "intent_analysis": intent_analysis,
                "tv_guide_knowledge": tv_guide_knowledge,
                "weather_lookup_notes": weather_facts,
                "failure_notes": failure_notes,
                "vlm_notes": vlm_hint,
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
    if config.enable_type_filter:
        if config.only_human_interaction:
            type_filter = set(config.test_types_to_include or [])
            type_filter.add("human_interaction")
            filtered_cases = [
                tc for tc in filtered_cases if any(t in type_filter for t in tc.types)
            ]
        elif config.test_types_to_include:
            filtered_cases = [
                tc
                for tc in filtered_cases
                if any(t in config.test_types_to_include for t in tc.types)
            ]

    if config.focus_test_names:
        focus_set = {
            name.strip()
            for name in config.focus_test_names
            if isinstance(name, str) and name.strip()
        }
        if focus_set:
            filtered_cases = [
                tc for tc in filtered_cases if tc.name in focus_set
            ]

    if config.max_test_cases is not None and config.max_test_cases > 0:
        filtered_cases = filtered_cases[: config.max_test_cases]

    repeat_runs = max(1, config.repeat_runs_per_test)
    if repeat_runs > 1:
        expanded_cases: List[TestCaseInfo] = []
        for tc in filtered_cases:
            expanded_cases.extend([tc] * repeat_runs)
        filtered_cases = expanded_cases
    
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
            planner_max_repeat_queries=config.planner_max_repeat_queries,
            planner_max_failures_per_action=config.planner_max_failures_per_action,
            enable_environment_overview=config.enable_environment_overview,
            enable_intent_analysis=config.enable_intent_analysis,
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
                "planner_max_repeat_queries": config.planner_max_repeat_queries,
                "planner_max_failures_per_action": config.planner_max_failures_per_action,
                "enable_environment_overview": config.enable_environment_overview,
                "enable_intent_analysis": config.enable_intent_analysis,
                "focus_test_names": config.focus_test_names,
                "repeat_runs_per_test": config.repeat_runs_per_test,
                "only_human_interaction": config.only_human_interaction,
                "enable_type_filter": config.enable_type_filter,
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



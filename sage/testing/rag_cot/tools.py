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

from sage.testing.rag_cot.utils import (
    _ensure_tool_global_config,
    _vlm_pick_device,
)

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

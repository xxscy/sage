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

TV_GUIDE_PATH = Path(__file__).parent.parent.joinpath("tv_guide.csv")
_TV_GUIDE_CACHE: Optional[List[Dict[str, str]]] = None

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


def _get_state_capture_test_config() -> Any:
    """
    构造与 test_runner 行为一致的最小配置，确保 coordinator_config 可用。
    该配置会被缓存复用，避免重复构造开销。
    """
    from types import SimpleNamespace
    from dataclasses import dataclass
    
    @dataclass
    class _LightweightTestConfig:
        """仿照 test_runner 中的 TestDemoConfig，仅保留评估所需的最小字段。"""
        coordinator_config: Any
        evaluator_llm: Any
    
    global _STATE_CAPTURE_TEST_CONFIG
    if '_STATE_CAPTURE_TEST_CONFIG' in globals() and _STATE_CAPTURE_TEST_CONFIG is not None:
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


def _prepare_device_state_for_test(test_case: Any) -> Dict[str, Any]:
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


def _collect_vlm_candidates(
    *, categories: List[str], device_state: Dict[str, Any], doc_manager: Optional[DocManager]
) -> List[str]:
    """仅收集与目标类别相关且有对应图片的设备 ID，用于 VLM 消歧。"""
    image_folder = Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
        "sage/testing/assets/images"
    )
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


# 全局变量
_STATE_CAPTURE_TEST_CONFIG: Optional[Any] = None


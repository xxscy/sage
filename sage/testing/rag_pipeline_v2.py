"""
Lightweight RAG + intent decision pipeline for testcase-driven evaluation.

Key behaviors:
- Load testcases from `sage/testing/testcases.py`.
- Initialize device_state per testcase before retrieval (reuses rag_cot_chain helpers).
- Summarization-only tools: preference lookup, device lookup, CLIP disambiguation,
  weather lookup, TV guide lookup, environment summary.
- Single reasoning step: intent analysis prompt (English).
- Final gate: decide Need / Do not need human_interaction_tool.

Assumptions:
- Device metadata via vector DB endpoint (optional). If DEVICE_VECTOR_API is unset,
  fallback to heuristic search on device_state only.
- CLIP disambiguation uses existing test images in `sage/testing/assets/images`.
- Memory bank at `data/memory_data/memory_bank.json`.
- TV guide from `sage/testing/tv_guide.csv` (inject-only style).
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from langchain.schema.messages import HumanMessage

from sage.retrieval.tools import UserProfileTool, UserProfileToolConfig
from sage.testing.rag_cot_chain import (
    _prepare_device_state_for_test,
    extract_user_command_from_test,
)
from sage.testing import testcases as testcase_module
from sage.utils.llm_utils import GPTConfig, LLMConfig
from sage.misc_tools.weather_tool import WeatherTool, WeatherToolConfig
from sage.smartthings.device_disambiguation import VlmDeviceDetector


# ------------------------------
# Data structures
# ------------------------------
@dataclass
class TestcaseRecord:
    name: str
    user_command: str
    types: List[str]
    requires_human_interaction: bool


@dataclass
class DeviceCandidate:
    device_id: str
    device_type: str
    name: Optional[str]
    state_excerpt: str
    score: float = 0.0


# ------------------------------
# Testcase loader & state init
# ------------------------------
class TestcaseLoader:
    def load(self) -> List[TestcaseRecord]:
        all_tests = testcase_module.get_tests(
            list(testcase_module.TEST_REGISTER.keys()), combination="union"
        )
        records: List[TestcaseRecord] = []
        for fn in all_tests:
            cmd = extract_user_command_from_test(fn)
            if not cmd:
                continue
            types = list(testcase_module.TEST_CASE_TYPES.get(fn.__name__, []))
            records.append(
                TestcaseRecord(
                    name=fn.__name__,
                    user_command=cmd,
                    types=types,
                    requires_human_interaction="human_interaction" in types,
                )
            )
        return records


class DeviceStateInitializer:
    def prepare(self, record: TestcaseRecord) -> Dict[str, Any]:
        class _Tmp:
            def __init__(self, name: str):
                self.name = name
                self._prepared_state = None

        tmp_case = _Tmp(record.name)
        return _prepare_device_state_for_test(tmp_case)  # deep copy inside helper


# ------------------------------
# Preference lookup
# ------------------------------
class PreferenceLookup:
    def __init__(self, memory_path: Optional[str] = None, llm_config: Optional[LLMConfig] = None):
        cfg = UserProfileToolConfig(
            memory_path=memory_path
            or f"{os.getenv('SMARTHOME_ROOT', '.')}/data/memory_data/memory_bank.json",
            llm_config=llm_config or GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False),
        )
        self.tool = cfg.instantiate()

    def run(self, user_name: str, query: Optional[str] = None) -> Dict[str, Any]:
        payload = json.dumps({"query": query or "Retrieve preferences and past interactions", "user_name": user_name})
        result = self.tool.run(payload)
        # tool returns string; keep raw
        return {"raw": result}


# ------------------------------
# Device lookup via vector DB + heuristics
# ------------------------------
DEVICE_CATEGORY_KEYWORDS = {
    "light": ("light", "lamp", "sconce", "bed", "fireplace", "dining", "tv light"),
    "tv": ("tv", "television", "screen", "channel", "volume"),
    "dishwasher": ("dishwasher", "dishes"),
    "fridge": ("fridge", "refrigerator", "freezer"),
}


def _infer_category(cmd: str) -> List[str]:
    lowered = cmd.lower()
    cats = []
    for cat, kws in DEVICE_CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            cats.append(cat)
    return cats


def _summarize_state(device_state: Dict[str, Any], device_id: str, max_components: int = 2) -> str:
    comp = device_state.get(device_id, {})
    parts: List[str] = []
    for idx, (comp_name, caps) in enumerate(comp.items()):
        if idx >= max_components:
            parts.append("...")
            break
        cap_parts = []
        for jdx, (cap_name, attrs) in enumerate(caps.items()):
            if jdx >= 3:
                cap_parts.append("...")
                break
            if isinstance(attrs, dict):
                # grab first 2 attrs
                inner = []
                for kdx, (attr_name, attr_val) in enumerate(attrs.items()):
                    if kdx >= 2:
                        inner.append("...")
                        break
                    val = attr_val["value"] if isinstance(attr_val, dict) and "value" in attr_val else attr_val
                    inner.append(f"{attr_name}={val}")
                cap_parts.append(f"{cap_name}: {'; '.join(inner)}")
            else:
                cap_parts.append(f"{cap_name}: {attrs}")
        parts.append(f"{comp_name} -> {' | '.join(cap_parts)}")
    return " || ".join(parts) if parts else "(no state)"


class DeviceVectorClient:
    """
    Simple HTTP client to a device vector DB.
    Expected env: DEVICE_VECTOR_API=http(s)://host:port/search
    Payload: {"query": "<string>", "limit": <int>}
    Returns: {"results":[{"device_id":..., "name":..., "type":..., "score":...}]}
    """

    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("DEVICE_VECTOR_API", "")

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        if not self.endpoint:
            return []
        try:
            resp = requests.post(self.endpoint, json={"query": query, "limit": limit}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", []) if isinstance(data, dict) else []
        except Exception:
            return []


class DeviceLookup:
    def __init__(self, image_folder: Optional[Path] = None):
        self.vector_client = DeviceVectorClient()
        self.image_folder = image_folder or Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
            "sage/testing/assets/images"
        )
        self.clip_detector = None
        if self.image_folder.exists():
            self.clip_detector = VlmDeviceDetector(str(self.image_folder))

    def _clip_disambiguate(self, devices: List[str], info: str) -> Optional[str]:
        if not self.clip_detector or not devices:
            return None
        try:
            payload = {"devices": devices, "disambiguation_information": info}
            winner = self.clip_detector.identify_device(json.dumps(payload))
            return winner if winner in devices else None
        except Exception:
            return None

    def lookup(self, user_command: str, device_state: Dict[str, Any], max_results: int = 3) -> Dict[str, Any]:
        categories = _infer_category(user_command)
        query = user_command
        remote_results = self.vector_client.search(query, limit=max_results)

        candidates: List[DeviceCandidate] = []
        if remote_results:
            for item in remote_results:
                device_id = item.get("device_id") or ""
                name = item.get("name")
                dtype = item.get("type") or ""
                score = float(item.get("score", 0.0))
                excerpt = _summarize_state(device_state, device_id)
                candidates.append(
                    DeviceCandidate(device_id=device_id, device_type=dtype, name=name, state_excerpt=excerpt, score=score)
                )
        else:
            # fallback heuristic: use device_state keys filtered by categories
            for dev_id in device_state.keys():
                dtype = "light" if "light" in dev_id else "tv" if "tv" in dev_id else "device"
                if categories and dtype not in categories:
                    continue
                excerpt = _summarize_state(device_state, dev_id)
                candidates.append(DeviceCandidate(device_id=dev_id, device_type=dtype, name=None, state_excerpt=excerpt))

        # CLIP disambiguation if multiple same-type
        by_type: Dict[str, List[DeviceCandidate]] = {}
        for c in candidates:
            by_type.setdefault(c.device_type, []).append(c)
        disamb_notes: List[str] = []
        final_candidates: List[DeviceCandidate] = []
        for dtype, group in by_type.items():
            if len(group) == 1:
                final_candidates.extend(group)
                continue
            ids = [g.device_id for g in group]
            winner = self._clip_disambiguate(ids, user_command)
            if winner:
                chosen = [g for g in group if g.device_id == winner][0]
                disamb_notes.append(f"CLIP picked {winner} for type {dtype}")
                final_candidates.append(chosen)
            else:
                final_candidates.extend(group[:max_results])

        final_candidates = sorted(final_candidates, key=lambda x: -x.score)[:max_results]
        return {
            "candidates": [
                {
                    "device_id": c.device_id,
                    "type": c.device_type,
                    "name": c.name,
                    "state_excerpt": c.state_excerpt,
                    "score": c.score,
                }
                for c in final_candidates
            ],
            "notes": "; ".join(disamb_notes) if disamb_notes else "(no disambiguation)",
        }


# ------------------------------
# Weather & TV
# ------------------------------
class WeatherLookup:
    def __init__(self, default_location: str = "quebec city, Canada"):
        self.cfg = WeatherToolConfig()
        self.tool = self.cfg.instantiate()
        self.default_location = default_location or "quebec city, Canada"

    def run(self, location: Optional[str]) -> str:
        loc = (location or self.default_location).strip() or self.default_location
        return self.tool.run(loc)


class TvGuideLookup:
    def __init__(self, tv_path: Optional[Path] = None):
        self.tv_path = tv_path or Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
            "sage/testing/tv_guide.csv"
        )
        self.rows = self._load()

    def _load(self) -> List[Dict[str, str]]:
        if not self.tv_path.exists():
            return []
        with self.tv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def lookup(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
        scored: List[Tuple[float, Dict[str, str]]] = []
        for row in self.rows:
            hay = " ".join(
                [
                    row.get("channel_name", ""),
                    row.get("program_name", ""),
                    row.get("program_desc", ""),
                ]
            ).lower()
            score = sum(1 for t in q_tokens if t and t in hay)
            scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k] if r]


# ------------------------------
# Summaries, intent, decision
# ------------------------------
def build_environment_summary(payload: Dict[str, Any]) -> str:
    devices = payload.get("devices", [])
    device_lines = [
        f"- {d.get('device_id')} ({d.get('type')}): {d.get('state_excerpt')}"
        for d in devices
    ] or ["(no devices)"]
    prefs = payload.get("preferences", "(no preferences)")
    history = payload.get("history_snippets", [])
    weather = payload.get("weather", "(no weather)")
    tv = payload.get("tv", [])
    tv_lines = [
        f"{t.get('channel_number')} {t.get('channel_name')} -> {t.get('program_name')}"
        for t in tv
    ] or ["(no tv results)"]
    parts = [
        "Devices: " + " | ".join(device_lines),
        "Preferences/Profile: " + str(prefs),
        "History: " + " || ".join(history) if history else "History: (none)",
        "Weather: " + str(weather),
        "TV: " + " | ".join(tv_lines),
    ]
    return "\n".join(parts)


INTENT_ANALYSIS_PROMPT = """Your task: produce a structured Refined Command.
Facts:
- Command: "{user_command}"
- Device Context: {device_context}
- Preferences/Profile: {preferences}
- History Snippets: {history}
- Weather: {weather}
- TV Guide: {tv}
- Environment Summary: {env_summary}

Rules:
- Content Consumption: only use verified source; otherwise set UNKNOWN_CHANNEL/UNKNOWN_SOURCE.
- Device Control: if parameters missing and unsafe to infer, set UNKNOWN_PARAMETER; otherwise apply safe defaults.
- Use disambiguated device_ids when provided.

Output:
- Request Type: <Device Control | Content Consumption>
- Identified Situation: <brief>
- Content Verification: <match found | no verifiable match>
- Refined Command: <explicit command with device_ids, UNKNOWN_* if needed>
- Confidence: <High | Low>
- Reasoning: <step-by-step>
"""


FINAL_DECISION_PROMPT = """Decide if clarification is required.
Input:
- User Command: "{user_command}"
- Refined Command: "{refined_command}"
- Content Verification: "{content_verification}"

Rules:
- If Content Consumption and source is unverified/UNKNOWN -> Need.
- If Device Control and required parameters cannot be safely defaulted -> Need.
- Otherwise -> Do not need.

Format:
Reasoning: Step 1 - … Step 2 - … Step 3 - … Step 4 - Final Verdict
Conclusion: Need to use human_interaction_tool | Do not need to use human_interaction_tool
Reason: <brief>
"""


class IntentAnalyzer:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = (llm_config or GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False)).instantiate()

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = INTENT_ANALYSIS_PROMPT.format(
            user_command=payload["user_command"],
            device_context=payload["device_context"],
            preferences=payload.get("preferences"),
            history="\n".join(payload.get("history_snippets", [])),
            weather=payload.get("weather"),
            tv=payload.get("tv"),
            env_summary=payload.get("env_summary"),
        )
        resp = self.llm([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"raw": text}


class DecisionMaker:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = (llm_config or GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False)).instantiate()

    def run(self, user_command: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        # naive extraction; this is intentionally simple
        refined = intent_result.get("raw", "")
        content_verif = "UNKNOWN" if "UNKNOWN" in refined else "match found"
        prompt = FINAL_DECISION_PROMPT.format(
            user_command=user_command,
            refined_command=refined,
            content_verification=content_verif,
        )
        resp = self.llm([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        needs_tool = "Need to use human_interaction_tool" in text
        return {"raw": text, "needs_tool": needs_tool}


# ------------------------------
# Planner (LLM chooses tools)
# ------------------------------
PLANNER_PROMPT = """You are orchestrating tool calls to gather only the necessary context for a smart-home command.
Decide the next action. Avoid repeating the same query. Stop when you have enough info and go to final_decision.

User command: "{user_command}"

Already retrieved:
- Preferences: {pref_status}
- Devices: {device_status}
- Weather: {weather_status}
- TV: {tv_status}

Actions (one per step):
- preference_lookup (needs: query string; include user intent)
- device_lookup (needs: query string; describe device/goal)
- weather_lookup (optional query "City, Country"; omit to use default)
- tv_lookup (needs: query; TV content)
- final_decision (no query; move to intent + decision prompts)

Respond JSON only:
{{"thought": "<brief>", "action": "<one of the above>", "query": "<string or null>"}}
"""


def parse_planner_response(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text.strip())
        action = obj.get("action", "final_decision")
        if action not in {
            "preference_lookup",
            "device_lookup",
            "weather_lookup",
            "tv_lookup",
            "final_decision",
        }:
            action = "final_decision"
        return {"thought": obj.get("thought", ""), "action": action, "query": obj.get("query")}
    except Exception:
        return {"thought": text.strip(), "action": "final_decision", "query": None}


class Planner:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = (llm_config or GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False)).instantiate()

    def run(self, user_command: str, state: Dict[str, Any]) -> Dict[str, Any]:
        pref_status = "available" if state.get("preferences") else "missing"
        device_status = "available" if state.get("devices") else "missing"
        weather_status = "available" if state.get("weather") else "missing"
        tv_status = "available" if state.get("tv") else "missing"
        prompt = PLANNER_PROMPT.format(
            user_command=user_command,
            pref_status=pref_status,
            device_status=device_status,
            weather_status=weather_status,
            tv_status=tv_status,
        )
        resp = self.llm([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return parse_planner_response(text)


# ------------------------------
# Orchestrator
# ------------------------------
@dataclass
class PipelineConfig:
    llm_config: LLMConfig = field(default_factory=lambda: GPTConfig(model_name="gpt-4o-mini", temperature=0.0, streaming=False))
    tv_top_k: int = 5
    device_max_results: int = 3
    default_location: str = "quebec city, Canada"
    log_dir: Optional[str] = str(Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath("test_logs", "rag_pipeline_v2"))


class RagPipeline:
    def __init__(self, config: PipelineConfig):
        # Default log dir under test_logs/rag_pipeline_v2 if not provided
        if config.log_dir is None:
            root = Path(os.getenv("SMARTHOME_ROOT", "."))
            config.log_dir = str(root.joinpath("test_logs", "rag_pipeline_v2"))
        self.config = config
        self.loader = TestcaseLoader()
        self.state_init = DeviceStateInitializer()
        self.pref_tool = PreferenceLookup(llm_config=config.llm_config)
        self.device_tool = DeviceLookup()
        self.weather_tool = WeatherLookup()
        self.tv_tool = TvGuideLookup()
        self.intent = IntentAnalyzer(llm_config=config.llm_config)
        self.decision = DecisionMaker(llm_config=config.llm_config)
        self.planner = Planner(llm_config=config.llm_config)

    def run_one(self, record: TestcaseRecord) -> Dict[str, Any]:
        # Display basic info for the testcase
        print(f"[Testcase] name={record.name} | types={record.types} | requires_human_interaction={record.requires_human_interaction}")
        print(f"[Command] {record.user_command}")

        device_state = self.state_init.prepare(record)
        user_name = (record.user_command.split(":")[0] or "").strip().lower()

        # planner-driven loop
        state: Dict[str, Any] = {"preferences": None, "devices": None, "weather": None, "tv": None}
        planner_history: List[Dict[str, Any]] = []
        halted_actions: set[str] = set()

        def _snapshot(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                return str(obj)

        for _ in range(12):
            step = self.planner.run(record.user_command, state)
            planner_history.append(step)
            action = step.get("action")
            query = step.get("query")
            if action in halted_actions:
                # action already proved unproductive; force exit
                break
            if action == "preference_lookup":
                before = _snapshot(state.get("preferences"))
                state["preferences"] = self.pref_tool.run(user_name=user_name, query=query)
                after = _snapshot(state.get("preferences"))
                if before == after:
                    halted_actions.add(action)
                continue
            if action == "device_lookup":
                q = query or record.user_command
                before = _snapshot(state.get("devices"))
                state["devices"] = self.device_tool.lookup(
                    q, device_state, max_results=self.config.device_max_results
                )
                after = _snapshot(state.get("devices"))
                if before == after:
                    halted_actions.add(action)
                continue
            if action == "weather_lookup":
                loc = (query or self.config.default_location).strip()
                before = _snapshot(state.get("weather"))
                state["weather"] = self.weather_tool.run(loc)
                after = _snapshot(state.get("weather"))
                if before == after:
                    halted_actions.add(action)
                continue
            if action == "tv_lookup":
                q = query or record.user_command
                before = _snapshot(state.get("tv"))
                state["tv"] = self.tv_tool.lookup(q, top_k=self.config.tv_top_k)
                after = _snapshot(state.get("tv"))
                if before == after:
                    halted_actions.add(action)
                continue
            if action == "final_decision":
                break

        env_summary = build_environment_summary(
            {
                "devices": (state.get("devices") or {}).get("candidates", [])
                if isinstance(state.get("devices"), dict)
                else (state.get("devices") or []),
                "preferences": (state.get("preferences") or {}).get("raw")
                if isinstance(state.get("preferences"), dict)
                else state.get("preferences"),
                "history_snippets": [],
                "weather": state.get("weather"),
                "tv": state.get("tv") or [],
            }
        )

        intent_input = {
            "user_command": record.user_command,
            "device_context": state.get("devices"),
            "preferences": (state.get("preferences") or {}).get("raw")
            if isinstance(state.get("preferences"), dict)
            else state.get("preferences"),
            "history_snippets": [],
            "weather": state.get("weather"),
            "tv": state.get("tv") or [],
            "env_summary": env_summary,
        }
        intent_res = self.intent.run(intent_input)
        decision_res = self.decision.run(record.user_command, intent_res)
        predicted = bool(decision_res.get("needs_tool"))
        ground = bool(record.requires_human_interaction)
        print(f"[Result] predicted_needs_tool={predicted} | ground_truth={ground} | correct={predicted == ground}")

        return {
            "test_name": record.name,
            "user_command": record.user_command,
            "types": record.types,
            "requires_human_interaction": record.requires_human_interaction,
            "device_state_init": "(omitted for brevity)",
            "preferences": state.get("preferences"),
            "device_lookup": state.get("devices"),
            "weather": state.get("weather"),
            "tv": state.get("tv"),
            "environment_summary": env_summary,
            "intent": intent_res,
            "decision": decision_res,
            "planner_history": planner_history,
        }

    def run_all(self) -> List[Dict[str, Any]]:
        results = []
        excluded_types = {"google", "test_id"}
        for rec in self.loader.load():
            if any(t in excluded_types for t in rec.types):
                continue
            results.append(self.run_one(rec))
        summary = self._summarize(results)
        self._write_logs(summary)
        return summary

    def _summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(results)
        correct = 0
        help_total = help_correct = 0
        non_help_total = non_help_correct = 0
        type_stats: Dict[str, Dict[str, int]] = {}

        for r in results:
            predicted = bool(r.get("decision", {}).get("needs_tool"))
            ground = bool(r.get("requires_human_interaction"))
            is_correct = predicted == ground
            r["is_correct"] = is_correct
            if is_correct:
                correct += 1
            if ground:
                help_total += 1
                if predicted:
                    help_correct += 1
            else:
                non_help_total += 1
                if not predicted:
                    non_help_correct += 1

            for t in r.get("types", []):
                stats = type_stats.setdefault(t, {"total": 0, "correct": 0})
                stats["total"] += 1
                if is_correct:
                    stats["correct"] += 1

        def _safe_ratio(num: int, den: int) -> float:
            return num / den if den else 0.0

        type_summary = {
            k: {
                "accuracy": _safe_ratio(v["correct"], v["total"]),
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in type_stats.items()
        }

        return {
            "total_cases": total,
            "correct": correct,
            "accuracy": _safe_ratio(correct, total),
            "help_total": help_total,
            "help_correct": help_correct,
            "help_accuracy": _safe_ratio(help_correct, help_total),
            "non_help_total": non_help_total,
            "non_help_correct": non_help_correct,
            "non_help_accuracy": _safe_ratio(non_help_correct, non_help_total),
            "type_statistics": type_summary,
            "results": results,
        }

    def _write_logs(self, summary: Dict[str, Any]) -> None:
        log_dir = self.config.log_dir
        if not log_dir:
            return
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            path = Path(log_dir).joinpath(ts)
            path.mkdir(parents=True, exist_ok=True)
            (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            # per-test logs
            for r in summary.get("results", []):
                fname = f"{r.get('test_name','unknown')}.json"
                (path / fname).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")

            # concise summary overview (human-readable)
            overview_lines = [
                f"Total cases: {summary.get('total_cases', 0)}",
                f"Correct: {summary.get('correct', 0)}",
                f"Accuracy: {summary.get('accuracy', 0):.2%}",
                f"Help (need) accuracy: {summary.get('help_accuracy', 0):.2%} "
                f"({summary.get('help_correct', 0)}/{summary.get('help_total', 0)})",
                f"Non-help accuracy: {summary.get('non_help_accuracy', 0):.2%} "
                f"({summary.get('non_help_correct', 0)}/{summary.get('non_help_total', 0)})",
                "Per-type accuracy:",
            ]
            type_stats = summary.get("type_statistics", {})
            if not type_stats:
                overview_lines.append("  (no type stats)")
            else:
                for k, v in type_stats.items():
                    overview_lines.append(
                        f"  {k}: {v.get('correct',0)}/{v.get('total',0)} "
                        f"({v.get('accuracy',0):.2%})"
                    )
            (path / "summary_overview.txt").write_text("\n".join(overview_lines), encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    pipeline = RagPipeline(PipelineConfig())
    summary = pipeline.run_all()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


"""
Hybrid test runner that combines rag_cot_chain guidance with the original
test_runner execution pipeline. The flow:

1. Use RAG-based tooling to analyze each testcase command up front and
   predict whether human_interaction_tool is needed (plus provide intent /
   environment summaries).
2. Feed the guidance into the downstream coordinator so that it only enables
   human interaction when RAG believes it is necessary (with optional runtime
   override when execution uncovers new ambiguities).
3. Run the original testcase just like test_runner.py, but log both the RAG
   judgment and the real execution result for later comparison.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from copy import deepcopy
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tyro

from sage.base import BaseConfig, GlobalConfig
from sage.utils.common import CONSOLE
from sage.testing.testcases import get_tests, TEST_REGISTER, TEST_CASE_TYPES
from sage.testing.testing_utils import (
    current_save_dir,
    get_base_device_state,
    get_min_device_state,
)

from sage.testing.test_runner import (
    merge_test_types,
    CoordinatorType,
    LlmType,
    TestDemoConfig,
)

from sage.testing import rag_cot_chain as rag


@dataclass
class HybridTestConfig(TestDemoConfig):
    """Extend the original TestDemoConfig with RAG guidance options."""

    def _default_rag_config() -> rag.RAGCOTConfig:
        cfg = rag.RAGCOTConfig()
        cfg.test_types_to_include = []
        return cfg

    rag_config: rag.RAGCOTConfig = field(default_factory=_default_rag_config)
    enforce_guidance: bool = True  # True => disable HI when RAG says no
    allow_runtime_override: bool = False  # coordinator may re-enable HI if execution fails
    guidance_log_subdir: str = "test_logs/rag_guided"


class RagGuidanceEngine:
    """Thin wrapper around rag_cot_chain evaluation for per-testcase predictions."""

    def __init__(self, rag_config: rag.RAGCOTConfig):
        self.config = rag_config
        if getattr(self.config.llm_config, "streaming", False):
            self.config.llm_config.streaming = False
        self.llm = self.config.llm_config.instantiate()

        memory_path = f"{os.getenv('SMARTHOME_ROOT')}/data/memory_data/memory_bank.json"
        user_profile_cfg = rag.UserProfileToolConfig(
            llm_config=self.config.llm_config,
            memory_path=memory_path,
        )
        user_profile_cfg.force_rebuild_index = True
        self.user_profile_tool = user_profile_cfg.instantiate()
        self.context_tool = rag.ContextUnderstandingTool()

        doc_root = Path(os.getenv("SMARTHOME_ROOT", "."))
        docmanager_cache_path = doc_root.joinpath(
            "external_api_docs", "cached_test_docmanager.json"
        )
        self.device_lookup_tool = rag.DeviceLookupTool(
            docmanager_cache_path=docmanager_cache_path,
            max_results=self.config.device_lookup_max_results,
            planner_llm_config=deepcopy(self.config.llm_config),
        )
        self.weather_lookup_tool = rag.WeatherLookupTool(
            default_location=self.config.default_weather_location
        )

    def predict(
        self,
        test_case: rag.TestCaseInfo,
        device_state_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run rag_cot evaluation for a single testcase using the provided state."""

        test_case._prepared_state = deepcopy(device_state_snapshot)
        self.device_lookup_tool.update_device_state(device_state_snapshot)
        result = rag.evaluate_test_case(
            test_case=test_case,
            llm=self.llm,
            user_profile_tool=self.user_profile_tool,
            context_tool=self.context_tool,
            device_lookup_tool=self.device_lookup_tool,
            preference_query_template=self.config.preference_query_template,
            log_base_dir=None,
            weather_lookup_tool=self.weather_lookup_tool,
        )
        return result


def _load_rag_case_map() -> Dict[str, rag.TestCaseInfo]:
    """Map testcase name to rag.TestCaseInfo for quick lookup."""
    case_map: Dict[str, rag.TestCaseInfo] = {}
    for tc in rag.load_test_cases():
        case_map[tc.name] = tc
    return case_map


def _clone_config_with_hi_policy(
    base_config: HybridTestConfig,
    enable_human_interaction: bool,
) -> HybridTestConfig:
    cloned = deepcopy(base_config)
    coord = cloned.coordinator_config
    if hasattr(coord, "enable_human_interaction"):
        coord.enable_human_interaction = enable_human_interaction
    return cloned


def main(config: HybridTestConfig):
    config.print_to_terminal()

    save_dir = Path(os.getenv("SMARTHOME_ROOT")).joinpath(config.logpath)
    os.makedirs(save_dir, exist_ok=True)
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    save_detail_dir = save_dir.joinpath(now_str)
    os.makedirs(save_detail_dir)
    current_save_dir[0] = save_detail_dir
    CONSOLE.log(f"[yellow]Saving logs in {save_detail_dir}")
    config.save(save_detail_dir)

    guidance_root = Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
        config.guidance_log_subdir, now_str
    )
    guidance_root.mkdir(parents=True, exist_ok=True)
    per_case_dir = guidance_root.joinpath("cases")
    per_case_dir.mkdir(parents=True, exist_ok=True)

    log_path = save_detail_dir.joinpath(now_str + ".json")
    test_log: Dict[str, Any] = {}

    if config.wandb_tracing:
        os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
        os.environ["WANDB_PROJECT"] = "langchain-tracing"

    condition_server_url = None
    for name, url in config.trigger_servers:
        if name == "condition":
            condition_server_url = url

    BaseConfig.global_config = GlobalConfig(
        condition_server_url=condition_server_url,
        docmanager_cache_path=Path(os.getenv("SMARTHOME_ROOT")).joinpath(
            "external_api_docs/cached_test_docmanager.json"
        ),
    )

    rag_engine = RagGuidanceEngine(config.rag_config)
    rag_case_map = _load_rag_case_map()

    if config.test_scenario == "in-dist":
        test_cases = list(
            set(get_tests(list(TEST_REGISTER.keys()), combination="union"))
            - set(get_tests(["test_set"]))
        )
    else:
        test_cases = get_tests(["test_set"])

    if not config.include_human_interaction:
        human_cases = get_tests(["human_interaction"])
        test_cases = list(set(test_cases) - set(human_cases))

    if not config.enable_google:
        google_cases = get_tests(["google"])
        test_cases = list(set(test_cases) - set(google_cases))

    guidance_records = []

    for case_func in test_cases:
        case_name = case_func.__name__
        try:
            CONSOLE.print(f"Starting : {case_func}")

            if isinstance(config.coordinator_config, rag.SAGECoordinatorConfig):
                device_state = deepcopy(get_base_device_state())
            else:
                device_state = deepcopy(get_min_device_state())

            rag_case = rag_case_map.get(case_name)
            guidance = None
            if rag_case is not None:
                guidance = rag_engine.predict(rag_case, deepcopy(device_state))
                guidance_records.append(
                    {
                        "case": case_name,
                        "prediction": guidance["predicted"],
                        "reasoning": guidance["reasoning"],
                        "types": guidance["types"],
                    }
                )

            need_human_interaction = (
                guidance["predicted"] if guidance is not None else True
            )
            if not config.enforce_guidance:
                need_human_interaction = True

            case_config = _clone_config_with_hi_policy(
                config,
                enable_human_interaction=need_human_interaction,
            )

            BaseConfig.global_config.current_test_case = case_name
            BaseConfig.global_config.current_test_types = list(
                TEST_CASE_TYPES.get(case_name, [])
            )
            BaseConfig.global_config.human_interaction_stats = {"success": 0, "failure": 0}

            start_time = time.time()
            case_func(device_state, case_config)
            duration = time.time() - start_time

            test_log[case_name] = {
                "case": case_name,
                "result": "success",
                "runtime": duration,
                "rag_prediction": guidance["predicted"] if guidance else None,
                "rag_reasoning": guidance["reasoning"] if guidance else None,
            }
            CONSOLE.log(f"[green]\ncase {case_name} WIN  \U0001F603")

        except Exception as exc:
            traceback.print_exc()
            test_log[case_name] = {
                "case": case_name,
                "result": "failure",
                "error": str(exc),
                "rag_prediction": guidance["predicted"] if guidance else None,
                "rag_reasoning": guidance["reasoning"] if guidance else None,
            }
            CONSOLE.log(f"[red]\ncase {case_name} Fail \U0001F914")
        finally:
            stats = getattr(BaseConfig.global_config, "human_interaction_stats", None)
            if (stats is not None) and (case_name in test_log):
                test_log[case_name]["human_interaction_tool_calls"] = dict(stats)

            BaseConfig.global_config.current_test_case = None
            BaseConfig.global_config.current_test_types = []
            BaseConfig.global_config.human_interaction_stats = {"success": 0, "failure": 0}

        merge_test_types(test_log)
        with open(log_path, "w", encoding="utf-8") as fp:
            json.dump(test_log, fp, ensure_ascii=False, indent=2)

        case_types = list(TEST_CASE_TYPES.get(case_name, []))
        ground_truth_hi = None
        if guidance and "ground_truth" in guidance:
            ground_truth_hi = guidance["ground_truth"]
        elif "human_interaction" in case_types:
            ground_truth_hi = True

        case_summary = {
            "test_name": case_name,
            "types": case_types,
            "requires_human_interaction_ground_truth": ground_truth_hi,
            "rag_prediction": guidance["predicted"] if guidance else None,
            "rag_reasoning": guidance["reasoning"] if guidance else None,
            "human_interaction_enabled": need_human_interaction,
            "execution_result": test_log[case_name]["result"],
            "error": test_log[case_name].get("error"),
            "human_interaction_tool_calls": test_log[case_name].get(
                "human_interaction_tool_calls", {}
            ),
        }

        safe_case_name = re.sub(r"[^\w.-]", "_", case_name)
        per_case_path = per_case_dir.joinpath(f"{safe_case_name}.json")
        per_case_path.write_text(
            json.dumps(case_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    merge_test_types(test_log)
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(test_log, fp, ensure_ascii=False, indent=2)

    guidance_path = guidance_root.joinpath("guidance_summary.json")
    guidance_path.write_text(
        json.dumps(guidance_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    CONSOLE.print("DONE (hybrid runner)!")
    success_rate = len([t for t in test_log.values() if t["result"] == "success"]) / len(
        test_log
    )
    CONSOLE.log("Success rate: ", success_rate)


if __name__ == "__main__":
    main(tyro.cli(HybridTestConfig))


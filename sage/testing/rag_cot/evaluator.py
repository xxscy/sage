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
    build_tv_guide_knowledge,
    extract_user_name_from_command,
)

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
            effective_user_name=effective_user_name,
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
            effective_user_name=effective_user_name,
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

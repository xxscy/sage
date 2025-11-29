"""
Contextual RAG COT evaluation without ReAct actions.

This module deterministically收集与测试用例相关的上下文信息
（用户偏好、设备状态、天气、环境摘要等），然后一次性把整理好的内容
交给 LLM 产出：
    1. 用户意图判断
    2. 与指令最相关的环境概览（自然语言）
    3. 最终推理过程 + 是否需要 human_interaction_tool 的结论

不需要 Thought/Action 结构，也不让 LLM 自主调用工具，便于稳定复现与日志分析。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.schema.messages import HumanMessage

from sage.utils.common import CONSOLE
from sage.testing import rag_cot_chain as legacy


WEATHER_KEYWORDS = {
    "weather",
    "rain",
    "snow",
    "umbrella",
    "sunny",
    "cloud",
    "temperature",
    "cold",
    "hot",
    "too cold",
    "too hot",
    "下雨",
    "天气",
    "带伞",
    "气温",
    "温度",
    "太热",
    "太冷",
}


@dataclass
class ContextSnapshot:
    user_preferences: Any = field(default_factory=dict)
    user_memory_snippets: List[str] = field(default_factory=list)
    device_lookup_notes: List[str] = field(default_factory=list)
    weather_reports: List[str] = field(default_factory=list)
    context_summary: Optional[str] = None
    environment_overview: Optional[str] = None
    intent_analysis: Optional[str] = None
    device_state_focus: str = ""
    device_state_summary: str = ""
    target_device_context: str = ""
    target_device_categories: List[str] = field(default_factory=list)
    tv_guide_knowledge: str = ""
    final_reasoning: Optional[str] = None
    predicted_needs_tool: Optional[bool] = None


def _should_request_weather(command: str) -> bool:
    lowered = command.lower()
    return any(keyword in lowered for keyword in WEATHER_KEYWORDS)


def _collect_context_snapshot(
    *,
    test_case: legacy.TestCaseInfo,
    llm,
    user_profile_tool,
    context_tool: legacy.ContextUnderstandingTool,
    device_lookup_tool: legacy.DeviceLookupTool,
    weather_lookup_tool: Optional[legacy.WeatherLookupTool],
    effective_user_name: str,
    preference_query: str,
) -> ContextSnapshot:
    """Gather preferences, device facts, weather, summaries, etc."""

    snapshot = ContextSnapshot()

    device_state = legacy._prepare_device_state_for_test(test_case)
    snapshot.device_state_summary = legacy._summarize_device_state(device_state)
    device_lookup_tool.update_device_state(device_state)
    doc_manager = getattr(device_lookup_tool, "doc_manager", None)
    snapshot.device_state_focus = legacy._build_device_state_focus(
        device_state, doc_manager
    )
    categories, target_context = legacy._build_target_device_context(
        command=test_case.user_command,
        device_state=device_state,
        doc_manager=doc_manager,
    )
    snapshot.target_device_categories = categories
    snapshot.target_device_context = target_context
    snapshot.tv_guide_knowledge = legacy.build_tv_guide_knowledge(
        test_case.user_command
    )

    try:
        pref_input = json.dumps(
            {"query": preference_query, "user_name": effective_user_name},
            ensure_ascii=False,
        )
        snapshot.user_preferences = user_profile_tool.run(pref_input)
    except Exception as exc:
        snapshot.user_preferences = f"UserProfileTool error: {exc}"

    try:
        device_fact = device_lookup_tool.run(test_case.user_command)
    except Exception as exc:
        device_fact = f"DeviceLookupTool error: {exc}"
    snapshot.device_lookup_notes.append(device_fact)

    if weather_lookup_tool and _should_request_weather(test_case.user_command):
        try:
            snapshot.weather_reports.append(
                weather_lookup_tool.run(weather_lookup_tool.default_location)
            )
        except Exception as exc:
            snapshot.weather_reports.append(f"WeatherLookupTool error: {exc}")

    snapshot.context_summary = context_tool.run(
        user_command=test_case.user_command,
        user_preferences=snapshot.user_preferences,
        device_state=device_state,
        user_memory_snippets=snapshot.user_memory_snippets,
        device_lookup_notes=snapshot.device_lookup_notes,
    )

    env_prompt = legacy.build_environment_overview_prompt(
        test_case=test_case,
        device_lookup_notes=snapshot.device_lookup_notes,
        device_state=device_state,
        target_device_context=snapshot.target_device_context,
    )
    env_resp = llm([HumanMessage(content=env_prompt)])
    snapshot.environment_overview = (
        env_resp.content if hasattr(env_resp, "content") else str(env_resp)
    )

    intent_prompt = legacy.build_intent_analysis_prompt(
        test_case=test_case,
        user_preferences=snapshot.user_preferences,
        user_memory_snippets=snapshot.user_memory_snippets,
        context_summary=snapshot.context_summary,
        device_lookup_notes=snapshot.device_lookup_notes,
        environment_overview=snapshot.environment_overview,
        tv_guide_knowledge=snapshot.tv_guide_knowledge,
        weather_reports=snapshot.weather_reports,
        device_state_focus=snapshot.device_state_focus,
        target_device_context=snapshot.target_device_context,
    )
    intent_resp = llm([HumanMessage(content=intent_prompt)])
    snapshot.intent_analysis = (
        intent_resp.content if hasattr(intent_resp, "content") else str(intent_resp)
    )

    return snapshot


def evaluate_test_case_contextual(
    test_case: legacy.TestCaseInfo,
    llm,
    user_profile_tool,
    context_tool: legacy.ContextUnderstandingTool,
    device_lookup_tool: legacy.DeviceLookupTool,
    preference_query_template: str,
    log_base_dir: Optional[Path] = None,
    weather_lookup_tool: Optional[legacy.WeatherLookupTool] = None,
) -> Dict[str, Any]:
    """Simplified contextual evaluation pipeline (no ReAct actions)."""

    preference_query = preference_query_template.format(
        user_command=test_case.user_command
    )

    parsed_user_name = legacy.extract_user_name_from_command(test_case.user_command)
    effective_user_name = parsed_user_name.strip() if parsed_user_name else ""
    if effective_user_name:
        CONSOLE.log(f"[cyan]从命令中提取到用户名: {effective_user_name}")
    else:
        CONSOLE.log(f"[yellow]警告: 无法从命令中提取用户名: {test_case.user_command}")

    CONSOLE.rule(f"[bold blue]测试用例: {test_case.name}")
    CONSOLE.log(f"[bold]用户指令[/bold]: {test_case.user_command}")

    snapshot = _collect_context_snapshot(
        test_case=test_case,
        llm=llm,
        user_profile_tool=user_profile_tool,
        context_tool=context_tool,
        device_lookup_tool=device_lookup_tool,
        weather_lookup_tool=weather_lookup_tool,
        effective_user_name=effective_user_name,
        preference_query=preference_query,
    )

    CONSOLE.log(f"[bold]设备状态摘要[/bold]: {snapshot.device_state_summary}")
    CONSOLE.log(f"[green]环境概览[/green]: {snapshot.environment_overview}")
    CONSOLE.log(f"[green]用户意图分析[/green]: {snapshot.intent_analysis}")

    final_prompt = legacy.build_cot_prompt(
        test_case=test_case,
        user_preferences=snapshot.user_preferences,
        user_memory_snippets=snapshot.user_memory_snippets,
        context_summary=snapshot.context_summary,
        device_lookup_notes=snapshot.device_lookup_notes,
        intent_analysis=snapshot.intent_analysis,
        environment_overview=snapshot.environment_overview,
        tv_guide_knowledge=snapshot.tv_guide_knowledge,
        weather_reports=snapshot.weather_reports,
        device_state_focus=snapshot.device_state_focus,
        target_device_context=snapshot.target_device_context,
    )
    final_resp = llm([HumanMessage(content=final_prompt)])
    final_text = final_resp.content if hasattr(final_resp, "content") else str(final_resp)
    needs_tool, reasoning = legacy.parse_llm_response(final_text)
    snapshot.predicted_needs_tool = needs_tool
    snapshot.final_reasoning = reasoning

    CONSOLE.log(
        f"[bold]预测是否需要 human_interaction_tool[/bold]: "
        f"{'需要' if needs_tool else '不需要'} "
        f"(期望: {'需要' if test_case.requires_human_interaction else '不需要'})"
    )
    CONSOLE.log("[bold]LLM 推理摘要[/bold]:")
    CONSOLE.log(reasoning)

    result: Dict[str, Any] = {
        "test_name": test_case.name,
        "user_command": test_case.user_command,
        "effective_user_name": effective_user_name,
        "types": test_case.types,
        "ground_truth": test_case.requires_human_interaction,
        "predicted": needs_tool,
        "is_correct": needs_tool == test_case.requires_human_interaction,
        "reasoning": reasoning,
        "llm_response": final_text,
        "user_preferences": snapshot.user_preferences,
        "device_lookup_notes": snapshot.device_lookup_notes,
        "weather_lookup_notes": snapshot.weather_reports,
        "context_summary": snapshot.context_summary,
        "environment_overview": snapshot.environment_overview,
        "intent_analysis": snapshot.intent_analysis,
        "device_state_focus": snapshot.device_state_focus,
        "device_state_summary": snapshot.device_state_summary,
        "target_device_context": snapshot.target_device_context,
        "target_device_categories": snapshot.target_device_categories,
        "tv_guide_knowledge": snapshot.tv_guide_knowledge,
    }

    if log_base_dir is not None:
        try:
            command_folder = legacy._sanitize_filename(test_case.user_command)
            case_dir = log_base_dir.joinpath(command_folder)
            case_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            log_path = case_dir.joinpath(f"{test_case.name}_{timestamp}.json")
            log_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            CONSOLE.log(f"[red]写入命令日志失败: {exc}")

    return result


def run_rag_cot_context_evaluation(
    config: Optional[legacy.RAGCOTConfig] = None,
) -> Dict[str, Any]:
    """Run the contextual evaluation loop."""

    if config is None:
        config = legacy.RAGCOTConfig()

    CONSOLE.log("[yellow]加载测试用例...")
    all_test_cases = legacy.load_test_cases()

    excluded_types = {"google", "test_set"}
    filtered_cases = [
        tc for tc in all_test_cases if not any(t in excluded_types for t in tc.types)
    ]

    if config.test_types_to_include:
        filtered_cases = [
            tc
            for tc in filtered_cases
            if any(t in config.test_types_to_include for t in tc.types)
        ]

    if config.max_test_cases:
        filtered_cases = filtered_cases[: config.max_test_cases]

    CONSOLE.log(f"[green]共加载 {len(filtered_cases)} 个测试用例")

    root_dir = Path(os.getenv("SMARTHOME_ROOT", "."))
    log_root = root_dir.joinpath("test_logs", "rag_cot_context")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_base_dir = log_root.joinpath(timestamp)
    log_base_dir.mkdir(parents=True, exist_ok=True)
    CONSOLE.log(f"[yellow]上下文模式日志将保存到: {log_base_dir}")

    if getattr(config.llm_config, "streaming", False):
        config.llm_config.streaming = False
    llm = config.llm_config.instantiate()

    memory_path = f"{os.getenv('SMARTHOME_ROOT')}/data/memory_data/memory_bank.json"
    user_profile_tool = legacy.UserProfileToolConfig(
        llm_config=config.llm_config,
        memory_path=memory_path,
    ).instantiate()

    context_tool = legacy.ContextUnderstandingTool()
    docmanager_cache_path = Path(os.getenv("SMARTHOME_ROOT", ".")).joinpath(
        "external_api_docs", "cached_test_docmanager.json"
    )
    device_lookup_tool = legacy.DeviceLookupTool(
        docmanager_cache_path=docmanager_cache_path,
        max_results=config.device_lookup_max_results,
        planner_llm_config=config.llm_config,
    )
    weather_lookup_tool = legacy.WeatherLookupTool(
        default_location=config.default_weather_location
    )

    results: List[Dict[str, Any]] = []
    correct_count = 0
    help_total = help_correct = 0
    non_help_total = non_help_correct = 0

    CONSOLE.log("[yellow]开始上下文模式评估...")
    for idx, test_case in enumerate(filtered_cases, 1):
        CONSOLE.log(f"[cyan]处理 {idx}/{len(filtered_cases)}: {test_case.name}")
        result = evaluate_test_case_contextual(
            test_case=test_case,
            llm=llm,
            user_profile_tool=user_profile_tool,
            context_tool=context_tool,
            device_lookup_tool=device_lookup_tool,
            preference_query_template=config.preference_query_template,
            log_base_dir=log_base_dir,
            weather_lookup_tool=weather_lookup_tool,
        )
        results.append(result)
        if result["is_correct"]:
            correct_count += 1

        if result["ground_truth"]:
            help_total += 1
            if result["predicted"]:
                help_correct += 1
        else:
            non_help_total += 1
            if not result["predicted"]:
                non_help_correct += 1

    accuracy = correct_count / len(filtered_cases) if filtered_cases else 0
    help_accuracy = help_correct / help_total if help_total else 0
    non_help_accuracy = (
        non_help_correct / non_help_total if non_help_total else 0
    )

    type_stats: Dict[str, Dict[str, int]] = {}
    for item in results:
        for test_type in item["types"]:
            type_stats.setdefault(test_type, {"total": 0, "correct": 0})
            type_stats[test_type]["total"] += 1
            if item["is_correct"]:
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
                "accuracy": v["correct"] / v["total"] if v["total"] else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in type_stats.items()
        },
        "results": results,
    }

    try:
        final_log_path = log_base_dir.joinpath("result_summary.json")
        final_log_path.write_text(
            json.dumps(
                {
                    "generated_at": timestamp,
                    "config": {
                        "user_name": config.user_name,
                        "max_test_cases": config.max_test_cases,
                        "test_types_to_include": config.test_types_to_include,
                        "device_lookup_max_results": config.device_lookup_max_results,
                        "llm_model": getattr(config.llm_config, "model_name", "unknown"),
                    },
                    "summary": summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        CONSOLE.log(f"[green]上下文模式汇总日志已写入: {final_log_path}")
    except Exception as exc:
        CONSOLE.log(f"[red]写入上下文模式汇总日志失败: {exc}")

    return summary


def print_context_evaluation_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print contextual evaluation statistics."""

    CONSOLE.rule("[bold green]上下文模式评估结果摘要")
    CONSOLE.log(f"总测试用例数: {summary['total_cases']}")
    CONSOLE.log(f"正确预测数: {summary['correct']}")
    CONSOLE.log(f"准确率: {summary['accuracy']:.2%}")
    CONSOLE.log(
        f"需要求助场景准确率: {summary['help_accuracy']:.2%} "
        f"({summary['help_correct']}/{summary['help_total']})"
    )
    CONSOLE.log(
        f"不需求助场景准确率: {summary['non_help_accuracy']:.2%} "
        f"({summary['non_help_correct']}/{summary['non_help_total']})"
    )

    CONSOLE.rule("[bold yellow]按类型统计")
    for test_type, stats in summary["type_statistics"].items():
        CONSOLE.log(
            f"{test_type}: {stats['correct']}/{stats['total']} "
            f"({stats['accuracy']:.2%})"
        )

    errors = [r for r in summary["results"] if not r["is_correct"]]
    if errors:
        CONSOLE.rule("[bold red]样例错误")
        for case in errors[:10]:
            CONSOLE.log(f"测试: {case['test_name']}")
            CONSOLE.log(f"  命令: {case['user_command']}")
            CONSOLE.log(
                f"  期望: {'需要' if case['ground_truth'] else '不需要'} | "
                f"预测: {'需要' if case['predicted'] else '不需要'}"
            )




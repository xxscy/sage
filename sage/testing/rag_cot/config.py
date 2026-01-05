"""
RAG COT 链配置模块
"""
from dataclasses import dataclass
from typing import Optional, List

from sage.utils.llm_utils import GPTConfig, LLMConfig


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












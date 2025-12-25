"""
RAG COT 链模块

将原来的 rag_cot_chain.py 拆分为多个模块，便于维护和扩展。
"""

from sage.testing.rag_cot.config import RAGCOTConfig
from sage.testing.rag_cot.tools import (
    ContextUnderstandingTool,
    DeviceLookupTool,
    WeatherLookupTool,
)
from sage.testing.rag_cot.testcase_loader import TestCaseInfo, load_test_cases
from sage.testing.rag_cot.evaluator import (
    evaluate_test_case,
    run_rag_cot_evaluation,
    print_evaluation_summary,
)

__all__ = [
    "RAGCOTConfig",
    "ContextUnderstandingTool",
    "DeviceLookupTool",
    "WeatherLookupTool",
    "TestCaseInfo",
    "load_test_cases",
    "evaluate_test_case",
    "run_rag_cot_evaluation",
    "print_evaluation_summary",
]


"""
RAG COT 链主入口

这是重构后的主入口文件，调用各个模块完成评估任务。
"""
from sage.testing.rag_cot.config import RAGCOTConfig
from sage.utils.llm_utils import GPTConfig


def main():
    """主入口函数"""
    # 暂时从原文件导入，待拆分完成后改为从新模块导入
    try:
        from sage.testing.rag_cot.evaluator import run_rag_cot_evaluation, print_evaluation_summary
    except ImportError:
        # 向后兼容：如果新模块不存在，从原文件导入
        from sage.testing.rag_cot_chain import run_rag_cot_evaluation, print_evaluation_summary
    
    # 示例使用
    config = RAGCOTConfig(
        llm_config=GPTConfig(model_name="gpt-4o-mini", temperature=0.0)
    )
    
    summary = run_rag_cot_evaluation(config)
    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()


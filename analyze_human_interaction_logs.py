#!/usr/bin/env python3
"""
分析测试日志中human_interaction_tool的调用情况
生成包含参数名称、类别、调用原因等的表格
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

def extract_reason(reasoning_text: str) -> str:
    """从reasoning文本中提取调用原因"""
    if not reasoning_text:
        return ""
    
    # 提取Reason部分
    reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', reasoning_text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
        # 清理markdown格式
        reason = re.sub(r'```', '', reason)
        reason = re.sub(r'\n+', ' ', reason)
        return reason[:200]  # 限制长度
    
    # 如果没有找到Reason，尝试提取Conclusion后的内容
    conclusion_match = re.search(r'Conclusion:.*?Reason:\s*(.+?)(?:\n|$)', reasoning_text, re.IGNORECASE | re.DOTALL)
    if conclusion_match:
        reason = conclusion_match.group(1).strip()
        reason = re.sub(r'```', '', reason)
        reason = re.sub(r'\n+', ' ', reason)
        return reason[:200]
    
    return ""

def extract_parameter_info(reasoning_text: str, user_command: str) -> Dict[str, Any]:
    """提取参数信息：缺失信息、模糊术语、设备类型、动作类型等"""
    params = {
        "missing_info": [],
        "ambiguous_terms": [],
        "device_type": "",
        "action_type": "",
        "parameter_categories": []
    }
    
    if not reasoning_text:
        return params
    
    # 提取缺失信息
    missing_patterns = [
        r'lacks?\s+(?:specific\s+)?(?:information|details?|data)\s+(?:about|on|regarding)\s+([^\.]+)',
        r'no\s+(?:information|data|details?)\s+(?:pertaining\s+to|about|on|regarding)\s+([^\.]+)',
        r'cannot\s+be\s+inferred\s+(?:from|about)\s+([^\.]+)',
        r'does\s+not\s+specify\s+([^\.]+)',
    ]
    
    for pattern in missing_patterns:
        matches = re.findall(pattern, reasoning_text, re.IGNORECASE)
        params["missing_info"].extend([m.strip() for m in matches if m.strip()])
    
    # 提取模糊术语
    ambiguous_patterns = [
        r'ambiguous\s+(?:reference|term|wording|pronoun)\s+(?:to|regarding)?\s*["\']?([^"\']+)["\']?',
        r'["\']([^"\']+)["\']\s+is\s+ambiguous',
        r'contains?\s+ambiguous\s+([^\.]+)',
    ]
    
    for pattern in ambiguous_patterns:
        matches = re.findall(pattern, reasoning_text, re.IGNORECASE)
        params["ambiguous_terms"].extend([m.strip() for m in matches if m.strip()])
    
    # 从命令中提取设备类型
    device_keywords = ['TV', 'light', 'dishwasher', 'fridge', 'refrigerator', 'fireplace', 'bedroom', 'dining room', 'living room']
    for keyword in device_keywords:
        if keyword.lower() in user_command.lower():
            params["device_type"] = keyword
            break
    
    # 从命令中提取动作类型
    action_keywords = ['turn on', 'turn off', 'set', 'adjust', 'change', 'put', 'play', 'dim', 'make']
    for keyword in action_keywords:
        if keyword.lower() in user_command.lower():
            params["action_type"] = keyword
            break
    
    # 参数类别分类
    categories = []
    if params["missing_info"]:
        categories.append("缺失信息")
    if params["ambiguous_terms"]:
        categories.append("模糊术语")
    if "preference" in reasoning_text.lower() or "favourite" in reasoning_text.lower() or "favorite" in reasoning_text.lower():
        categories.append("用户偏好")
    if "device" in reasoning_text.lower() and "ambiguous" in reasoning_text.lower():
        categories.append("设备歧义")
    if "pronoun" in reasoning_text.lower() or "it" in user_command.lower() or "this" in user_command.lower():
        categories.append("代词歧义")
    
    params["parameter_categories"] = categories
    
    return params

def analyze_logs(log_file: Path) -> List[Dict[str, Any]]:
    """分析日志文件，提取所有调用human_interaction_tool的案例"""
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for case in data.get('summary', {}).get('results', []):
        # 检查是否预测需要human_interaction
        predicted = case.get('predicted', False) or case.get('predicted_requires_human_interaction', False)
        ground_truth = case.get('ground_truth', False) or case.get('ground_truth_requires_human_interaction', False)
        
        # 只分析预测为True的案例
        if predicted:
            reasoning = case.get('reasoning', '') or case.get('llm_response', '')
            reason = extract_reason(reasoning)
            params = extract_parameter_info(reasoning, case.get('user_command', ''))
            
            result = {
                "test_name": case.get('test_name', ''),
                "user_command": case.get('user_command', ''),
                "user_name": case.get('effective_user_name', ''),
                "types": ', '.join(case.get('types', [])),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": case.get('is_correct', False),
                "reason": reason,
                "missing_info": '; '.join(params['missing_info'][:3]) if params['missing_info'] else '',
                "ambiguous_terms": '; '.join(params['ambiguous_terms'][:3]) if params['ambiguous_terms'] else '',
                "device_type": params['device_type'],
                "action_type": params['action_type'],
                "parameter_categories": ', '.join(params['parameter_categories']) if params['parameter_categories'] else '',
                "reasoning_full": reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
            }
            results.append(result)
    
    return results

def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
    """生成Markdown表格"""
    if not results:
        return "未找到需要human_interaction的案例"
    
    # 表头
    headers = [
        "序号", "测试名称", "用户命令", "用户名", "测试类型", 
        "设备类型", "动作类型", "参数类别", "缺失信息", 
        "模糊术语", "调用原因", "是否正确"
    ]
    
    # 生成表格行
    rows = []
    for idx, result in enumerate(results, 1):
        row = [
            str(idx),
            result['test_name'],
            result['user_command'][:50] + "..." if len(result['user_command']) > 50 else result['user_command'],
            result['user_name'],
            result['types'],
            result['device_type'] or '-',
            result['action_type'] or '-',
            result['parameter_categories'] or '-',
            result['missing_info'][:50] + "..." if len(result['missing_info']) > 50 else (result['missing_info'] or '-'),
            result['ambiguous_terms'][:50] + "..." if len(result['ambiguous_terms']) > 50 else (result['ambiguous_terms'] or '-'),
            result['reason'][:80] + "..." if len(result['reason']) > 80 else (result['reason'] or '-'),
            "是" if result['is_correct'] else "否"
        ]
        rows.append(row)
    
    # 生成Markdown表格
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for row in rows:
        md_lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md_lines)

def generate_summary_statistics(results: List[Dict[str, Any]]) -> str:
    """生成统计摘要"""
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    incorrect = total - correct
    
    # 统计参数类别
    category_counts = defaultdict(int)
    for result in results:
        categories = result['parameter_categories'].split(', ') if result['parameter_categories'] else []
        for cat in categories:
            if cat:
                category_counts[cat] += 1
    
    # 统计设备类型
    device_counts = defaultdict(int)
    for result in results:
        if result['device_type']:
            device_counts[result['device_type']] += 1
    
    # 统计动作类型
    action_counts = defaultdict(int)
    for result in results:
        if result['action_type']:
            action_counts[result['action_type']] += 1
    
    summary = f"""## 统计摘要

### 总体统计
- **总案例数**: {total}
- **正确预测**: {correct} ({correct/total*100:.1f}%)
- **错误预测**: {incorrect} ({incorrect/total*100:.1f}%)

### 参数类别分布
"""
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"- **{cat}**: {count} 次\n"
    
    summary += "\n### 设备类型分布\n"
    for device, count in sorted(device_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"- **{device}**: {count} 次\n"
    
    summary += "\n### 动作类型分布\n"
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"- **{action}**: {count} 次\n"
    
    return summary

def main():
    log_file = Path("test_logs/rag_cot/2025-11-22_12-07-39_288258/result_summary.json")
    
    if not log_file.exists():
        print(f"日志文件不存在: {log_file}")
        return
    
    print("正在分析日志文件...")
    results = analyze_logs(log_file)
    
    print(f"找到 {len(results)} 个需要human_interaction的案例")
    
    # 生成Markdown报告
    report = "# Human Interaction Tool 调用分析报告\n\n"
    report += generate_summary_statistics(results)
    report += "\n\n## 详细表格\n\n"
    report += generate_markdown_table(results)
    
    # 添加调用原因分析
    report += "\n\n## LLM调用human_interaction_tool的主要原因\n\n"
    report += """
根据分析，LLM提出需要调用human_interaction_tool的主要原因包括：

### 1. **信息缺失 (Missing Information)**
- **描述**: 命令中缺少执行操作所需的关键信息
- **示例**: 
  - "adjust the volume" - 缺少具体音量值
  - "set an appropriate mode" - 缺少模式定义
  - "my favourite color" - 缺少颜色偏好信息

### 2. **模糊术语 (Ambiguous Terms)**
- **描述**: 命令中包含不明确或主观的术语
- **示例**:
  - "the game" - 未指定具体游戏
  - "redonkulous" - 非标准主观表达
  - "non essential devices" - 未定义哪些设备

### 3. **代词歧义 (Pronoun Ambiguity)**
- **描述**: 使用代词但上下文不明确
- **示例**:
  - "Turn it off" - "it"指代不明
  - "this channel" - "this"指代不明

### 4. **用户偏好缺失 (Missing User Preferences)**
- **描述**: 需要用户偏好但系统中不存在
- **示例**:
  - "my favourite color" - 偏好未存储
  - "cozy setting" - 偏好定义不明确

### 5. **设备歧义 (Device Ambiguity)**
- **描述**: 设备识别不明确
- **示例**:
  - "the TV" - 多个TV时指代不明
  - "the light" - 多个灯时指代不明

### 6. **非标准表达 (Non-standard Expressions)**
- **描述**: 使用非标准或创造性词汇
- **示例**:
  - "redonkulous" - 非标准词汇
  - "something light and entertaining" - 主观描述

### 调用决策流程

LLM通过以下步骤决定是否调用human_interaction_tool：

1. **关键词分析**: 识别命令中的动作和目标
2. **歧义检查**: 检测模糊术语、代词、非标准表达
3. **信息可用性检查**: 验证所需信息是否可从用户偏好、历史记录或设备状态中获取
4. **决策**: 如果关键信息缺失且无法推断，则调用human_interaction_tool

### 最佳实践建议

1. **增强上下文理解**: 利用对话历史解决代词歧义
2. **完善用户偏好库**: 存储更多用户偏好以减少询问
3. **改进设备识别**: 使用更智能的设备匹配算法
4. **处理非标准表达**: 建立同义词和上下文映射
5. **智能推断**: 在安全范围内进行合理推断，减少不必要的交互
"""
    
    # 保存报告
    output_file = Path("human_interaction_analysis_report.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n分析完成！报告已保存到: {output_file}")
    print(f"\n预览前3个案例:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {result['test_name']}")
        print(f"   命令: {result['user_command']}")
        print(f"   原因: {result['reason'][:100]}...")
        print(f"   正确: {'Y' if result['is_correct'] else 'N'}")

if __name__ == "__main__":
    main()


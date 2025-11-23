#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析日志中调用human_interaction_tool的案例
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any

def extract_reason(reasoning_text: str) -> str:
    """从reasoning文本中提取原因"""
    if not reasoning_text:
        return ""
    
    # 提取Reason字段
    reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', reasoning_text, re.MULTILINE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
        # 清理markdown代码块标记
        reason = reason.replace('```', '').strip()
        return reason
    
    # 如果没有找到Reason，尝试提取Conclusion后的内容
    conclusion_match = re.search(r'Conclusion:.*?\n(.*?)(?:\n\n|$)', reasoning_text, re.MULTILINE | re.DOTALL)
    if conclusion_match:
        reason = conclusion_match.group(1).strip()
        reason = reason.replace('```', '').strip()
        return reason
    
    return ""

def extract_parameter_info(reasoning_text: str, user_command: str) -> Dict[str, Any]:
    """提取参数信息"""
    params = {
        "missing_info": [],
        "ambiguous_terms": [],
        "device_type": "",
        "action_type": ""
    }
    
    # 提取缺失的信息
    missing_patterns = [
        r"lacks\s+(?:specific\s+)?information\s+about\s+([^,\.]+)",
        r"no\s+information\s+pertaining\s+to\s+([^,\.]+)",
        r"cannot\s+be\s+inferred\s+from\s+([^,\.]+)",
        r"requires?\s+clarification\s+on\s+([^,\.]+)",
    ]
    
    for pattern in missing_patterns:
        matches = re.findall(pattern, reasoning_text, re.IGNORECASE)
        params["missing_info"].extend([m.strip() for m in matches])
    
    # 提取模糊术语
    ambiguous_patterns = [
        r'\"([^\"]+)\"',
        r'appropriate\s+(\w+)',
        r'desired\s+(\w+)',
    ]
    
    for pattern in ambiguous_patterns:
        matches = re.findall(pattern, reasoning_text, re.IGNORECASE)
        params["ambiguous_terms"].extend([m.strip() for m in matches])
    
    # 提取设备类型
    device_keywords = ["TV", "dishwasher", "fridge", "refrigerator", "light", "lights", "fireplace"]
    for keyword in device_keywords:
        if keyword.lower() in user_command.lower():
            params["device_type"] = keyword
            break
    
    # 提取动作类型
    action_keywords = ["adjust", "set", "change", "turn on", "turn off", "dim", "play"]
    for keyword in action_keywords:
        if keyword.lower() in user_command.lower():
            params["action_type"] = keyword
            break
    
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
                "reasoning_full": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            }
            results.append(result)
    
    return results

def generate_markdown_table(results: List[Dict[str, Any]]) -> str:
    """生成Markdown表格"""
    headers = [
        "序号", "测试名称", "用户命令", "用户名", "测试类型", 
        "设备类型", "动作类型", "缺失信息", "模糊术语", 
        "调用原因", "是否正确"
    ]
    
    rows = []
    for i, r in enumerate(results, 1):
        row = [
            str(i),
            r['test_name'],
            r['user_command'][:50] + "..." if len(r['user_command']) > 50 else r['user_command'],
            r['user_name'],
            r['types'],
            r['device_type'],
            r['action_type'],
            r['missing_info'][:40] + "..." if len(r['missing_info']) > 40 else r['missing_info'],
            r['ambiguous_terms'][:40] + "..." if len(r['ambiguous_terms']) > 40 else r['ambiguous_terms'],
            r['reason'][:60] + "..." if len(r['reason']) > 60 else r['reason'],
            "✓" if r['is_correct'] else "✗"
        ]
        rows.append("| " + " | ".join(row) + " |")
    
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    return "\n".join([header_row, separator] + rows)

def main():
    log_file = Path("test_logs/rag_cot/2025-11-22_12-07-39_288258/result_summary.json")
    
    if not log_file.exists():
        print(f"日志文件不存在: {log_file}")
        return
    
    print("正在分析日志...")
    results = analyze_logs(log_file)
    
    print(f"\n找到 {len(results)} 个调用human_interaction_tool的案例\n")
    
    # 生成表格
    table = generate_markdown_table(results)
    
    # 保存到文件
    output_file = Path("human_interaction_analysis.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Human Interaction Tool 调用分析\n\n")
        f.write(f"## 总览\n\n")
        f.write(f"- 总案例数: {len(results)}\n")
        f.write(f"- 正确预测: {sum(1 for r in results if r['is_correct'])}\n")
        f.write(f"- 错误预测: {sum(1 for r in results if not r['is_correct'])}\n\n")
        
        f.write("## 详细表格\n\n")
        f.write(table)
        f.write("\n\n")
        
        f.write("## 调用原因分类\n\n")
        
        # 统计调用原因
        reasons = {}
        for r in results:
            reason_key = r['reason'][:50] if r['reason'] else "未指定"
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
        
        f.write("### 原因统计\n\n")
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{reason}**: {count} 次\n")
        
        f.write("\n### 设备类型分布\n\n")
        device_types = {}
        for r in results:
            device = r['device_type'] or "未指定"
            device_types[device] = device_types.get(device, 0) + 1
        
        for device, count in sorted(device_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{device}**: {count} 次\n")
        
        f.write("\n### 动作类型分布\n\n")
        action_types = {}
        for r in results:
            action = r['action_type'] or "未指定"
            action_types[action] = action_types.get(action, 0) + 1
        
        for action, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{action}**: {count} 次\n")
    
    print(f"分析结果已保存到: {output_file}")
    
    # 打印前5个案例
    print("\n前5个案例预览:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. {r['test_name']}")
        print(f"   命令: {r['user_command']}")
        print(f"   原因: {r['reason'][:100]}...")
        print(f"   缺失信息: {r['missing_info']}")

if __name__ == "__main__":
    main()


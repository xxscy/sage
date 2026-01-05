#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成测试结果表格
"""

import json
import csv
from pathlib import Path

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_markdown_table(data):
    """生成Markdown表格"""
    summary = data['summary']
    results = data['summary']['results']
    
    # 创建表格内容
    table_lines = []
    
    # 添加标题
    table_lines.append("# 测试结果汇总表\n")
    table_lines.append(f"**生成时间**: {data['generated_at']}\n")
    table_lines.append(f"**模型**: {data['config']['llm_model']}\n")
    table_lines.append(f"**总测试数**: {summary['total_cases']}\n")
    table_lines.append(f"**正确数**: {summary['correct']}\n")
    table_lines.append(f"**准确率**: {summary['accuracy']:.2%}\n\n")
    
    # 添加类型统计
    table_lines.append("## 类型统计\n")
    table_lines.append("| 类型 | 正确数 | 总数 | 准确率 |\n")
    table_lines.append("|------|--------|------|--------|\n")
    
    type_stats = summary['type_statistics']
    for type_name, stats in sorted(type_stats.items()):
        table_lines.append(f"| {type_name} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.2%} |\n")
    
    table_lines.append("\n")
    
    # 添加详细结果表格
    table_lines.append("## 详细测试结果\n")
    table_lines.append("| 序号 | 测试名称 | 用户命令 | 用户名 | 类型 | 是否正确 | 预测值 | 真实值 | 链式步骤数 |\n")
    table_lines.append("|------|----------|----------|--------|------|----------|--------|--------|------------|\n")
    
    for idx, result in enumerate(results, 1):
        test_name = result['test_name']
        user_command = result['user_command'].replace('|', '\\|')  # 转义管道符
        user_name = result['effective_user_name']
        types = ', '.join(result['types'])
        is_correct = '✓' if result['is_correct'] else '✗'
        predicted = 'ACT' if result['predicted'] else 'SKIP'
        ground_truth = 'ACT' if result['ground_truth'] else 'SKIP'
        chain_steps = len(result.get('chain_history', []))
        
        # 截断过长的命令
        if len(user_command) > 60:
            user_command = user_command[:57] + '...'
        
        table_lines.append(f"| {idx} | {test_name} | {user_command} | {user_name} | {types} | {is_correct} | {predicted} | {ground_truth} | {chain_steps} |\n")
    
    return ''.join(table_lines)

def generate_csv_table(data, output_path):
    """生成CSV表格"""
    results = data['summary']['results']
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            '序号', '测试名称', '用户命令', '用户名', '类型', 
            '是否正确', '预测值', '真实值', '链式步骤数', '推理'
        ])
        
        # 写入数据
        for idx, result in enumerate(results, 1):
            writer.writerow([
                idx,
                result['test_name'],
                result['user_command'],
                result['effective_user_name'],
                ', '.join(result['types']),
                '是' if result['is_correct'] else '否',
                'ACT' if result['predicted'] else 'SKIP',
                'ACT' if result['ground_truth'] else 'SKIP',
                len(result.get('chain_history', [])),
                result.get('reasoning', '')[:100]  # 限制长度
            ])

def generate_summary_csv(data, output_path):
    """生成汇总统计CSV"""
    summary = data['summary']
    type_stats = summary['type_statistics']
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 总体统计
        writer.writerow(['统计项', '数值'])
        writer.writerow(['总测试数', summary['total_cases']])
        writer.writerow(['正确数', summary['correct']])
        writer.writerow(['准确率', f"{summary['accuracy']:.2%}"])
        writer.writerow(['需要帮助总数', summary['help_total']])
        writer.writerow(['需要帮助正确数', summary['help_correct']])
        writer.writerow(['需要帮助准确率', f"{summary['help_accuracy']:.2%}"])
        writer.writerow(['不需要帮助总数', summary['non_help_total']])
        writer.writerow(['不需要帮助正确数', summary['non_help_correct']])
        writer.writerow(['不需要帮助准确率', f"{summary['non_help_accuracy']:.2%}"])
        writer.writerow([])
        
        # 类型统计
        writer.writerow(['类型', '正确数', '总数', '准确率'])
        for type_name, stats in sorted(type_stats.items()):
            writer.writerow([
                type_name,
                stats['correct'],
                stats['total'],
                f"{stats['accuracy']:.2%}"
            ])

def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    json_file = script_dir / 'result_summary.json'
    
    # 加载数据
    print(f"正在加载 {json_file}...")
    data = load_json(json_file)
    
    # 生成Markdown表格
    md_output = script_dir / 'test_results_table.md'
    print(f"正在生成Markdown表格: {md_output}")
    md_content = generate_markdown_table(data)
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    # 生成详细CSV表格
    csv_output = script_dir / 'test_results_detail.csv'
    print(f"正在生成详细CSV表格: {csv_output}")
    generate_csv_table(data, csv_output)
    
    # 生成汇总CSV表格
    summary_csv_output = script_dir / 'test_results_summary.csv'
    print(f"正在生成汇总CSV表格: {summary_csv_output}")
    generate_summary_csv(data, summary_csv_output)
    
    print("\n完成！已生成以下文件：")
    print(f"  - {md_output}")
    print(f"  - {csv_output}")
    print(f"  - {summary_csv_output}")

if __name__ == '__main__':
    main()


import json
import sys
from collections import defaultdict

# 设置输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 读取JSON文件
with open('test/2025-11-22_15-39-03_675774/2025-11-22_15-39-03_675774.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 统计变量
help_success = 0  # 求助成功
help_failure = 0  # 求助失败
no_help_success = 0  # 不求助成功
no_help_failure = 0  # 不求助失败

# 按types分类统计
type_stats = defaultdict(lambda: {
    'help_success': 0,
    'help_failure': 0,
    'no_help_success': 0,
    'no_help_failure': 0,
    'total': 0
})

# 遍历所有测试用例
for case_name, case_data in data.items():
    types = case_data.get('types', [])
    success = case_data.get('human_interaction_tool_calls', {}).get('success', 0)
    failure = case_data.get('human_interaction_tool_calls', {}).get('failure', 0)
    
    has_human_interaction = 'human_interaction' in types
    
    # 统计总体
    if has_human_interaction:
        if success == 1:
            help_success += 1
        else:
            help_failure += 1
    else:
        if failure == 0:
            no_help_success += 1
        else:
            no_help_failure += 1
    
    # 按types分类统计
    for type_name in types:
        type_stats[type_name]['total'] += 1
        if has_human_interaction:
            if success == 1:
                type_stats[type_name]['help_success'] += 1
            else:
                type_stats[type_name]['help_failure'] += 1
        else:
            if failure == 0:
                type_stats[type_name]['no_help_success'] += 1
            else:
                type_stats[type_name]['no_help_failure'] += 1

# 计算成功率
total_help = help_success + help_failure
total_no_help = no_help_success + no_help_failure

help_success_rate = (help_success / total_help * 100) if total_help > 0 else 0
no_help_success_rate = (no_help_success / total_no_help * 100) if total_no_help > 0 else 0

# 输出结果
print("=" * 80)
print("测试结果分析")
print("=" * 80)
print(f"\n总体统计:")
print(f"  求助成功: {help_success}")
print(f"  求助失败: {help_failure}")
print(f"  求助总数: {total_help}")
print(f"  求助成功率: {help_success_rate:.2f}%")
print(f"\n  不求助成功: {no_help_success}")
print(f"  不求助失败: {no_help_failure}")
print(f"  不求助总数: {total_no_help}")
print(f"  不求助成功率: {no_help_success_rate:.2f}%")

print("\n" + "=" * 80)
print("按types分类统计")
print("=" * 80)

# 按类型名称排序
for type_name in sorted(type_stats.keys()):
    stats = type_stats[type_name]
    total = stats['total']
    help_total = stats['help_success'] + stats['help_failure']
    no_help_total = stats['no_help_success'] + stats['no_help_failure']
    
    print(f"\n{type_name}:")
    print(f"  总测试数: {total}")
    
    if help_total > 0:
        help_rate = (stats['help_success'] / help_total * 100) if help_total > 0 else 0
        print(f"  求助成功: {stats['help_success']}, 求助失败: {stats['help_failure']}, 求助总数: {help_total}")
        print(f"  求助成功率: {help_rate:.2f}%")
    
    if no_help_total > 0:
        no_help_rate = (stats['no_help_success'] / no_help_total * 100) if no_help_total > 0 else 0
        print(f"  不求助成功: {stats['no_help_success']}, 不求助失败: {stats['no_help_failure']}, 不求助总数: {no_help_total}")
        print(f"  不求助成功率: {no_help_rate:.2f}%")


import json
from collections import defaultdict

# 读取JSON文件
with open('test/2025-11-20_17-47-51_158928/2025-11-20_17-47-51_158928.json', 'r', encoding='utf-8') as f:
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
    human_interaction_tool_calls = case_data.get('human_interaction_tool_calls', {})
    success = human_interaction_tool_calls.get('success', 0)
    failure = human_interaction_tool_calls.get('failure', 0)
    
    has_human_interaction = 'human_interaction' in types
    
    # 判断求助情况
    if has_human_interaction:
        if success == 1:
            help_success += 1
            for t in types:
                type_stats[t]['help_success'] += 1
                type_stats[t]['total'] += 1
        else:
            help_failure += 1
            for t in types:
                type_stats[t]['help_failure'] += 1
                type_stats[t]['total'] += 1
    else:
        if failure == 0:
            no_help_success += 1
            for t in types:
                type_stats[t]['no_help_success'] += 1
                type_stats[t]['total'] += 1
        else:
            no_help_failure += 1
            for t in types:
                type_stats[t]['no_help_failure'] += 1
                type_stats[t]['total'] += 1

# 计算成功率
help_total = help_success + help_failure
no_help_total = no_help_success + no_help_failure

help_success_rate = (help_success / help_total * 100) if help_total > 0 else 0
no_help_success_rate = (no_help_success / no_help_total * 100) if no_help_total > 0 else 0

# 打印结果
print("=" * 80)
print("总体统计")
print("=" * 80)
print(f"求助成功: {help_success}")
print(f"求助失败: {help_failure}")
print(f"求助总数: {help_total}")
print(f"求助成功率: {help_success_rate:.2f}%")
print()
print(f"不求助成功: {no_help_success}")
print(f"不求助失败: {no_help_failure}")
print(f"不求助总数: {no_help_total}")
print(f"不求助成功率: {no_help_success_rate:.2f}%")
print()

# 按types分类统计
print("=" * 80)
print("按types分类统计")
print("=" * 80)

# 按类型名称排序
sorted_types = sorted(type_stats.items())

for type_name, stats in sorted_types:
    type_help_total = stats['help_success'] + stats['help_failure']
    type_no_help_total = stats['no_help_success'] + stats['no_help_failure']
    
    type_help_rate = (stats['help_success'] / type_help_total * 100) if type_help_total > 0 else 0
    type_no_help_rate = (stats['no_help_success'] / type_no_help_total * 100) if type_no_help_total > 0 else 0
    
    print(f"\n类型: {type_name}")
    print(f"  总测试数: {stats['total']}")
    print(f"  求助成功: {stats['help_success']}, 求助失败: {stats['help_failure']}, 求助总数: {type_help_total}, 求助成功率: {type_help_rate:.2f}%")
    print(f"  不求助成功: {stats['no_help_success']}, 不求助失败: {stats['no_help_failure']}, 不求助总数: {type_no_help_total}, 不求助成功率: {type_no_help_rate:.2f}%")

print("\n" + "=" * 80)









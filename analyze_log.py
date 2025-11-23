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
print("Overall Statistics")
print("=" * 80)
print(f"Help Success: {help_success}")
print(f"Help Failure: {help_failure}")
print(f"Help Total: {help_total}")
print(f"Help Success Rate: {help_success_rate:.2f}%")
print()
print(f"No Help Success: {no_help_success}")
print(f"No Help Failure: {no_help_failure}")
print(f"No Help Total: {no_help_total}")
print(f"No Help Success Rate: {no_help_success_rate:.2f}%")
print()

# 按types分类统计
print("=" * 80)
print("Statistics by Types")
print("=" * 80)

# 按类型名称排序
sorted_types = sorted(type_stats.items())

for type_name, stats in sorted_types:
    type_help_total = stats['help_success'] + stats['help_failure']
    type_no_help_total = stats['no_help_success'] + stats['no_help_failure']
    
    type_help_rate = (stats['help_success'] / type_help_total * 100) if type_help_total > 0 else 0
    type_no_help_rate = (stats['no_help_success'] / type_no_help_total * 100) if type_no_help_total > 0 else 0
    
    print(f"\nType: {type_name}")
    print(f"  Total Tests: {stats['total']}")
    print(f"  Help Success: {stats['help_success']}, Help Failure: {stats['help_failure']}, Help Total: {type_help_total}, Help Success Rate: {type_help_rate:.2f}%")
    print(f"  No Help Success: {stats['no_help_success']}, No Help Failure: {stats['no_help_failure']}, No Help Total: {type_no_help_total}, No Help Success Rate: {type_no_help_rate:.2f}%")

print("\n" + "=" * 80)


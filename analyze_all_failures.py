import json
import re
from pathlib import Path
from collections import defaultdict

# 读取结果文件
result_file = Path('test_logs/rag_cot/2025-12-24_16-33-42_902090/result_summary.json')
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

failures = []
for r in data['summary']['results']:
    if not r.get('is_correct', False):
        failures.append(r)

print(f"失败案例总数: {len(failures)}\n")
print("=" * 80)

# 读取每个失败案例的详细日志
failure_patterns = defaultdict(list)

for failure in failures:
    test_name = failure['test_name']
    command = failure['user_command']
    ground_truth = failure.get('ground_truth', False)
    predicted = failure.get('predicted', False)
    intent_analysis = failure.get('intent_analysis', '')
    reasoning = failure.get('reasoning', '')
    
    # 提取关键信息
    refined_match = re.search(r"Refined Command[：:]\s*(.+?)(?:\n|$)", intent_analysis, re.MULTILINE | re.DOTALL)
    refined_cmd = refined_match.group(1).strip() if refined_match else "N/A"
    
    confidence_match = re.search(r"Confidence[：:]\s*(High|Low)", intent_analysis, re.IGNORECASE)
    confidence = confidence_match.group(1).capitalize() if confidence_match else "Unknown"
    
    has_ambiguous = "AMBIGUOUS_DEVICE" in refined_cmd
    has_unknown = "UNKNOWN_PARAMETER" in refined_cmd or "UNKNOWN_CHANNEL" in refined_cmd
    
    # 分析失败原因
    cmd_lower = command.lower()
    
    # 检查是否是信息查询
    is_query = any(word in cmd_lower for word in ['what', 'is', 'are', 'check', 'query', 'should', '?'])
    
    # 检查是否是集体引用
    is_collective = any(word in cmd_lower for word in ['all', 'every', 'entire', 'both', 'all the'])
    
    # 检查是否有位置信息
    has_location = any(word in cmd_lower for word in ['by the', 'over the', 'in the', 'same room', 'other'])
    
    # 检查是否是条件触发
    is_conditional = 'when' in cmd_lower or 'if' in cmd_lower
    
    failure_patterns[test_name] = {
        'command': command,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'intent_analysis': intent_analysis[:500],
        'reasoning': reasoning[:500],
        'refined_cmd': refined_cmd[:200],
        'confidence': confidence,
        'has_ambiguous': has_ambiguous,
        'has_unknown': has_unknown,
        'is_query': is_query,
        'is_collective': is_collective,
        'has_location': has_location,
        'is_conditional': is_conditional,
    }

# 按模式分类
print("\n=== 失败模式分类 ===\n")

query_failures = [f for f in failure_patterns.values() if f['is_query']]
collective_failures = [f for f in failure_patterns.values() if f['is_collective']]
location_failures = [f for f in failure_patterns.values() if f['has_location']]
conditional_failures = [f for f in failure_patterns.values() if f['is_conditional']]

print(f"1. 信息查询失败: {len(query_failures)} 个")
for f in query_failures[:3]:
    print(f"   - {f['command'][:60]}...")
    print(f"     Refined: {f['refined_cmd'][:100]}")
    print(f"     标记: AMBIGUOUS={f['has_ambiguous']}, UNKNOWN={f['has_unknown']}")

print(f"\n2. 集体引用失败: {len(collective_failures)} 个")
for f in collective_failures[:3]:
    print(f"   - {f['command'][:60]}...")
    print(f"     Refined: {f['refined_cmd'][:100]}")
    print(f"     标记: AMBIGUOUS={f['has_ambiguous']}, UNKNOWN={f['has_unknown']}")

print(f"\n3. 位置信息失败: {len(location_failures)} 个")
for f in location_failures[:3]:
    print(f"   - {f['command'][:60]}...")
    print(f"     Refined: {f['refined_cmd'][:100]}")
    print(f"     标记: AMBIGUOUS={f['has_ambiguous']}, UNKNOWN={f['has_unknown']}")

print(f"\n4. 条件触发失败: {len(conditional_failures)} 个")
for f in conditional_failures[:3]:
    print(f"   - {f['command'][:60]}...")
    print(f"     Refined: {f['refined_cmd'][:100]}")
    print(f"     标记: AMBIGUOUS={f['has_ambiguous']}, UNKNOWN={f['has_unknown']}")

# 输出所有失败案例供详细分析
print("\n=== 所有失败案例详情 ===\n")
for name, info in failure_patterns.items():
    print(f"\n{name}:")
    print(f"  命令: {info['command']}")
    print(f"  Refined Command: {info['refined_cmd']}")
    print(f"  置信度: {info['confidence']}")
    print(f"  标记: AMBIGUOUS={info['has_ambiguous']}, UNKNOWN={info['has_unknown']}")
    print(f"  意图分析片段: {info['intent_analysis'][:200]}...")








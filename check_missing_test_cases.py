"""检查哪些测试用例无法提取用户命令"""
import inspect
import re
from sage.testing.testcases import get_tests, TEST_CASE_TYPES
from sage.testing import testcases

def extract_user_command_from_test(test_func) -> str:
    """从测试函数中提取用户命令"""
    source = inspect.getsource(test_func)
    
    # 首先尝试查找 coordinator.execute("...") 或 coordinator.execute('...')
    execute_patterns = [
        r'coordinator\.execute\(\s*["\']([^"\']+)["\']\s*\)',
        r'coordinator\.execute\(\s*\n\s*["\']([^"\']+)["\']\s*\n\s*\)',
    ]
    
    for pattern in execute_patterns:
        matches = re.findall(pattern, source, re.MULTILINE | re.DOTALL)
        if matches:
            command = matches[0].strip()
            if command:
                return command
    
    # 如果没找到直接字符串，查找变量赋值
    variable_patterns = [
        r'user_command\s*=\s*["\']([^"\']+)["\']',
        r'command\s*=\s*["\']([^"\']+)["\']',
    ]
    
    for pattern in variable_patterns:
        match = re.search(pattern, source, re.DOTALL)
        if match:
            command = match.group(1).strip()
            if command:
                return command
    
    # 如果还是没找到，尝试查找 coordinator.execute() 调用，可能使用变量
    execute_match = re.search(r'coordinator\.execute\(([^)]+)\)', source, re.DOTALL)
    if execute_match:
        var_name = execute_match.group(1).strip()
        var_pattern = rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']'
        var_match = re.search(var_pattern, source, re.DOTALL)
        if var_match:
            return var_match.group(1).strip()
    
    return None

# 获取所有测试用例
all_tests = get_tests(list(testcases.TEST_REGISTER.keys()), combination="union")

print(f"总测试用例数: {len(all_tests)}")
print("=" * 80)

cases_with_command = []
cases_without_command = []

for test_func in all_tests:
    test_name = test_func.__name__
    types = list(TEST_CASE_TYPES.get(test_name, []))
    
    # 排除 google 和 test_set 类型
    if any(t in {"google", "test_set"} for t in types):
        continue
    
    user_command = extract_user_command_from_test(test_func)
    
    if user_command:
        cases_with_command.append((test_name, user_command))
    else:
        cases_without_command.append((test_name, types))

print(f"\n能提取命令的测试用例: {len(cases_with_command)}")
print(f"无法提取命令的测试用例: {len(cases_without_command)}")
print()

if cases_without_command:
    print("=" * 80)
    print("无法提取命令的测试用例:")
    print("=" * 80)
    # 创建一个名称到函数的映射
    test_func_map = {test_func.__name__: test_func for test_func in all_tests}
    
    for name, types in cases_without_command:
        print(f"\n{name}")
        print(f"  类型: {types}")
        # 显示源代码片段
        if name in test_func_map:
            test_func = test_func_map[name]
            source = inspect.getsource(test_func)
            # 查找 coordinator.execute 调用
            if 'coordinator.execute' in source:
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if 'coordinator.execute' in line:
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        print(f"  相关代码:")
                        for j in range(start, end):
                            marker = ">>> " if j == i else "    "
                            print(f"  {marker}{lines[j]}")
                        break

print("\n" + "=" * 80)
print("所有能提取命令的测试用例:")
print("=" * 80)
for name, cmd in sorted(cases_with_command):
    print(f"  {name}")


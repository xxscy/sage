"""检查哪些测试用例无法提取 user_command"""
import inspect
import re
from sage.testing.testcases import get_tests, TEST_CASE_TYPES, TEST_REGISTER

def extract_user_command_from_test(test_func) -> str:
    """
    从测试函数中提取用户命令
    
    查找 coordinator.execute() 调用中的字符串参数
    优先查找直接传入execute的字符串，其次查找变量赋值
    """
    source = inspect.getsource(test_func)
    
    # 首先尝试查找 coordinator.execute("...") 或 coordinator.execute('...')
    # 匹配带引号的字符串参数
    execute_patterns = [
        r'coordinator\.execute\(["\']([^"\']+)["\']\)',  # 单行字符串
        r'coordinator\.execute\(([^)]+)\)',  # 更宽泛的匹配，可能包含变量
    ]
    
    for pattern in execute_patterns:
        matches = re.findall(pattern, source)
        if matches:
            # 取第一个匹配的命令
            command = matches[0].strip()
            # 如果是变量名，跳过（后面会处理）
            if not (command.startswith('"') or command.startswith("'")):
                continue
            # 移除引号
            command = command.strip('"\'').strip()
            if command:  # 确保不是空字符串
                return command
    
    # 如果没找到直接字符串，查找变量赋值
    # 查找 user_command = "..." 或 command = "..."
    variable_patterns = [
        r'user_command\s*=\s*["\']([^"\']+)["\']',
        r'command\s*=\s*["\']([^"\']+)["\']',
    ]
    
    for pattern in variable_patterns:
        match = re.search(pattern, source)
        if match:
            command = match.group(1).strip()
            if command:
                return command
    
    # 如果还是没找到，尝试查找 coordinator.execute() 调用，可能使用变量
    # 这种情况下，我们尝试找到最近的字符串赋值
    execute_match = re.search(r'coordinator\.execute\(([^)]+)\)', source)
    if execute_match:
        var_name = execute_match.group(1).strip()
        # 尝试找到这个变量的赋值
        var_pattern = rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']'
        var_match = re.search(var_pattern, source)
        if var_match:
            return var_match.group(1).strip()
    
    return None

# 获取所有注册的测试函数
all_tests = get_tests(list(TEST_REGISTER.keys()), combination="union")

print(f"总测试用例数: {len(all_tests)}")

# 统计能提取和不能提取的测试用例
can_extract = []
cannot_extract = []

for test_func in all_tests:
    test_name = test_func.__name__
    types = list(TEST_CASE_TYPES.get(test_name, []))
    user_command = extract_user_command_from_test(test_func)
    
    # 排除 google 和 test_set 类型
    excluded_types = {"google", "test_set"}
    has_excluded = any(t in excluded_types for t in types)
    
    if user_command:
        can_extract.append((test_name, types, has_excluded))
    else:
        cannot_extract.append((test_name, types, has_excluded))

print(f"\n能提取 user_command 的测试用例: {len(can_extract)}")
print(f"不能提取 user_command 的测试用例: {len(cannot_extract)}")

# 统计排除 google 和 test_set 后的数量
filtered_can_extract = [tc for tc in can_extract if not tc[2]]
print(f"\n排除 google 和 test_set 后能提取的测试用例: {len(filtered_can_extract)}")

if cannot_extract:
    print("\n无法提取 user_command 的测试用例:")
    for name, types, has_excluded in cannot_extract:
        excluded_mark = " [EXCLUDED]" if has_excluded else ""
        print(f"  - {name}: {types}{excluded_mark}")

# 列出所有排除后的测试用例名称
print("\n排除 google 和 test_set 后的测试用例列表:")
for name, types, _ in filtered_can_extract:
    print(f"  - {name}: {types}")









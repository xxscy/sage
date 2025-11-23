"""找出无法提取用户命令的测试用例"""
import inspect
import re
from sage.testing.testcases import get_tests, TEST_CASE_TYPES

def extract_user_command_from_test(test_func) -> str:
    """
    从测试函数中提取用户命令
    
    查找 coordinator.execute() 调用中的字符串参数
    优先查找直接传入execute的字符串，其次查找变量赋值
    """
    source = inspect.getsource(test_func)
    
    # 首先尝试查找 coordinator.execute("...") 或 coordinator.execute('...')
    # 支持单行和多行字符串
    # 匹配模式：coordinator.execute("...") 或 coordinator.execute(\n    "..."\n)
    execute_patterns = [
        # 单行：coordinator.execute("...")
        r'coordinator\.execute\(\s*["\']([^"\']+)["\']\s*\)',
        # 多行：coordinator.execute(\n    "..."\n)
        r'coordinator\.execute\(\s*\n\s*["\']([^"\']+)["\']\s*\n\s*\)',
    ]
    
    for pattern in execute_patterns:
        matches = re.findall(pattern, source, re.MULTILINE | re.DOTALL)
        if matches:
            # 取第一个匹配的命令
            command = matches[0].strip()
            if command:  # 确保不是空字符串
                return command
    
    # 如果没找到直接字符串，查找变量赋值
    # 查找 user_command = "..." 或 command = "..."
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
    # 这种情况下，我们尝试找到最近的字符串赋值
    execute_match = re.search(r'coordinator\.execute\(([^)]+)\)', source, re.DOTALL)
    if execute_match:
        var_name = execute_match.group(1).strip()
        # 尝试找到这个变量的赋值
        var_pattern = rf'{re.escape(var_name)}\s*=\s*["\']([^"\']+)["\']'
        var_match = re.search(var_pattern, source, re.DOTALL)
        if var_match:
            return var_match.group(1).strip()
    
    return None


# 加载所有测试用例
from sage.testing import testcases
all_tests = get_tests(list(testcases.TEST_REGISTER.keys()), combination="union")

print(f"总共找到 {len(all_tests)} 个测试函数\n")

# 检查每个测试用例
missing_commands = []
has_commands = []

for test_func in all_tests:
    test_name = test_func.__name__
    user_command = extract_user_command_from_test(test_func)
    
    if not user_command:
        missing_commands.append(test_name)
        print(f"❌ {test_name}: 无法提取用户命令")
        # 打印源代码的前几行以便调试
        source = inspect.getsource(test_func)
        lines = source.split('\n')[:10]
        print(f"   源代码前10行:")
        for i, line in enumerate(lines, 1):
            print(f"   {i:2d}: {line}")
        print()
    else:
        has_commands.append(test_name)

print("=" * 80)
print(f"总结:")
print(f"  可以提取命令的测试用例: {len(has_commands)}")
print(f"  无法提取命令的测试用例: {len(missing_commands)}")
print(f"  总计: {len(all_tests)}")
print()
if missing_commands:
    print("无法提取命令的测试用例列表:")
    for name in sorted(missing_commands):
        print(f"  - {name}")




import json
import os
from pathlib import Path

# 读取JSON文件
json_path = 'test/2025-11-20_17-47-51_158928/2025-11-20_17-47-51_158928.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取JSON中的所有测试用例名称
json_cases = set(data.keys())
print(f"JSON file contains {len(json_cases)} test cases")
print()

# 获取文件夹中的所有目录（排除文件和非测试用例目录）
folder_path = Path('test/2025-11-20_17-47-51_158928')
folder_dirs = set()
excluded_dirs = set()  # 记录被排除的目录

for item in folder_path.iterdir():
    if item.is_dir():
        folder_dirs.add(item.name)
    elif item.is_file():
        excluded_dirs.add(item.name)

# 排除已知的配置文件和结果文件
known_files = {'coord_config.yaml', 'test_config.yaml', 'result_summary.json', 
               '2025-11-20_17-47-51_158928.json'}

print(f"Total directories in folder: {len(folder_dirs)}")
print(f"Excluded files: {sorted(excluded_dirs)}")
print()

# 找出差异
only_in_json = json_cases - folder_dirs
only_in_folder = folder_dirs - json_cases

print("=" * 80)
print("Comparison Results")
print("=" * 80)

print(f"\nJSON test cases: {len(json_cases)}")
print(f"Folder directories: {len(folder_dirs)}")

if only_in_json:
    print(f"\nOnly in JSON ({len(only_in_json)}):")
    for case in sorted(only_in_json):
        print(f"  - {case}")

if only_in_folder:
    print(f"\nOnly in folder ({len(only_in_folder)}):")
    for case in sorted(only_in_folder):
        print(f"  - {case}")

if not only_in_json and not only_in_folder:
    print("\nJSON and folder are perfectly matched!")
    print(f"Both contain {len(json_cases)} test cases.")

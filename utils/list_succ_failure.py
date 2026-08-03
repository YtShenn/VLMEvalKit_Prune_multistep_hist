import os
from collections import defaultdict

def count_idx_distribution(root_path):
    # 数据结构示例: { 'A': {'failure': [1, 2], 'success': [3, 4]}, 'B': ... }
    results = defaultdict(lambda: defaultdict(list))

    # 遍历主文件夹下的子文件夹 (A, B, C...)
    subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    print(f"Found subfolders: {subfolders}")

    for sub in subfolders:
        for status in ['failures', 'success']:
            target_path = os.path.join(root_path, sub, status)
            
            if os.path.exists(target_path):
                files = os.listdir(target_path)
                for file_name in files:
                    # 检查是否以 idx 开头
                    if file_name.startswith('idx'):
                        # 尝试提取 idx 后的数字部分
                        # 假设格式是 "idx123.jpg" 或 "idx_123.txt"
                        # 取出 'idx' 之后、第一个非数字字符之前的数字部分
                        num_part = ""
                        for char in file_name[3:]: # 跳过 'idx'
                            if char.isdigit():
                                num_part += char
                            else:
                                break
                        
                        if num_part and int(num_part) not in results[sub][status]:
                            results[sub][status].append(int(num_part))

    # 按照要求的格式输出
    for sub in sorted(results.keys()):
        print(f"{sub}:")
        for status in ['failures', 'success']:
            indices = results[sub][status]
            # 这里对数字进行排序，方便查看分布
            indices.sort()
            # 将列表转换为字符串显示
            dist_str = ", ".join(map(str, indices)) if indices else "None"
            print(f"  {status}: {dist_str}")
        print("-" * 20)

# 使用方法：将 'your_data_folder' 替换为你实际的文件夹路径
data_folder='/home/ytshen/storage_net2/VLMEvalKit_Prune3/visualize_token_attn/qwen3vl_spvlm_nonprune_bbox'
count_idx_distribution(data_folder)
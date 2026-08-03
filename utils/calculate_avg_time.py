import re

def calculate_average_timings(file_path):
    # 匹配 vision_s, llm_s, other_s 后面紧跟的浮点数
    pattern = re.compile(r"vision_s=([\d\.]+)\s+llm_s=([\d\.]+)\s+other_s=([\d\.]+)")
    
    v_total, l_total, o_total = 0.0, 0.0, 0.0
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                # 提取数值并累加
                v_total += float(match.group(1))
                l_total += float(match.group(2))
                o_total += float(match.group(3))
                count += 1

    if count == 0:
        print("未在文件中找到匹配的 [StageTiming] 数据行。")
        return

    # 计算平均值
    avg_v = v_total / count
    avg_l = l_total / count
    avg_o = o_total / count
    avg_all = (v_total + l_total + o_total) / count

    print(f"--- 统计结果 (样本数: {count}) ---")
    print(f"平均 Vision 时间 (s): {avg_v:.4f}")
    print(f"平均 LLM 时间    (s): {avg_l:.4f}")
    print(f"平均 Other 时间  (s): {avg_o:.4f}")
    print(f"---------------------------------")
    print(f"单样本平均总耗时  (s): {avg_all:.4f}")

def process_attn_times(file_path, group_size=17):
    # 匹配 [Attn Time]: 0.0670 s 格式中的数字
    pattern = re.compile(r"\[Attn Time\]:\s*([\d\.]+)\s*s")
    
    times = []
    
    # 1. 提取所有秒数
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                times.append(float(match.group(1)))
    
    if not times:
        print("未发现匹配的数据。")
        return

    # 2. 按 group_size (17) 组进行求和
    # 使用列表推导式，每隔 17 个取一个切片进行求和
    sums = [sum(times[i : i + group_size]) for i in range(0, len(times), group_size)]
    
    # 3. 计算这些“和”的平均值
    if sums:
        final_avg = sum(sums) / len(sums)
        
        print(f"--- 处理报告 ---")
        print(f"总数据行数: {len(times)}")
        print(f"分组数量 (每组 {group_size} 个): {len(sums)}")
        print(f"最后一组的样本数: {len(times) % group_size if len(times) % group_size != 0 else group_size}")
        print(f"----------------")
        print(f"每组之和的平均值: {final_avg:.6f} s")

def process_flexible_attn_groups(file_path):
    # 匹配 [Attn Time]: 0.0670 s
    attn_pattern = re.compile(r"\[Attn Time\]:\s*([\d\.]+)\s*s")
    # 匹配需要忽略的干扰行，比如 [Efficiency INFO]
    ignore_pattern = re.compile(r"\[Efficiency INFO\]")
    
    all_groups = []
    current_group = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            attn_match = attn_pattern.search(line)
            
            if attn_match:
                # 命中 Attn Time，加入当前组
                current_group.append(float(attn_match.group(1)))
            elif ignore_pattern.search(line):
                # 命中干扰行，跳过但不中断当前组
                continue
            else:
                # 命中了既不是 Attn 也不是 INFO 的行（即边界行）
                if current_group:
                    all_groups.append(current_group)
                    current_group = []
        
        # 处理文件末尾最后一组
        if current_group:
            all_groups.append(current_group)

    if not all_groups:
        print("未发现有效数据。")
        return

    # 计算每组的和，再求平均
    group_sums = [sum(g) for g in all_groups]
    final_avg = sum(group_sums) / len(group_sums)
    
    print(f"--- 灵活分组报告 ---")
    print(f"总计检测到组数: {len(all_groups)}")
    # 打印前几组的数量，方便你核对是否接近 17
    group_sizes = [len(g) for g in all_groups]
    print(f"每组包含的记录数（前10组）: {group_sizes[:10]}")
    print(f"最大组记录数: {max(group_sizes)}, 最小组记录数: {min(group_sizes)}")
    print(f"-------------------")
    print(f"所有组之和的平均值: {final_avg:.6f} s")

# 使用方法：替换为你的日志文件路径
# calculate_average_timings('run_output_20260320_170922_scale0.5.log')
process_flexible_attn_groups('run_output_20260321_223001.log')
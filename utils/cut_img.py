#!/usr/bin/env python3
"""
批量裁剪 screenspotpro 数据集中的图片
根据 ROI 坐标，为每张图片裁剪 5 个感兴趣区域（ROI）
"""

import json
import os
from pathlib import Path
from PIL import Image
import sys

# 配置参数
ROI_RESULTS_JSON = "/home/ytshen/storage_net2/VLMEvalKit_Prune_my/qwen3vl_1-15_layer_0.5scale_name/roi_results.json"
OUTPUT_DIR = "/home/ytshen/storage_net2/VLMEvalKit_Prune_my/roi_cropped_images_scale"

# 可能的图片目录（按优先级排列）
POSSIBLE_SCREENSHOT_DIRS = '/mnt/storage2/users/ytshen_data/LMUData/images/'

def find_image_directory():
    """找到包含图片的目录"""
    print("正在查找图片目录...")
    
    dir_path = POSSIBLE_SCREENSHOT_DIRS
    if os.path.exists(dir_path):
        # 检查目录中是否有 PNG 文件
        png_files = list(Path(dir_path).glob("**/*.png"))
        if png_files:
            print(f"✓ 找到图片目录: {dir_path}")
            return dir_path
    
    # 如果都没找到，尝试查找任何包含 attn.png 的目录
    print("在系统中搜索 attn.png 文件...")
    try:
        result = os.popen("find /mnt/storage2 -name '*attn.png' -type f 2>/dev/null | head -1").read().strip()
        if result:
            img_dir = os.path.dirname(result)
            print(f"✓ 找到图片目录: {img_dir}")
            return img_dir
    except:
        pass
    
    return None

def load_roi_results(json_path):
    """加载ROI结果JSON文件"""
    try:
        with open(json_path, 'r') as f:
            print(f"正在加载ROI结果: {json_path}")
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def crop_and_save_rois(roi_dict, screenshot_dir, output_dir):
    """
    根据ROI结果裁剪图片并保存
    
    Args:
        roi_dict: 从JSON加载的ROI字典 {图片名: [[x1,y1,x2,y2], ...]}
        screenshot_dir: 原始图片目录
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    total_cropped = 0
    total_images = len(roi_dict)
    processed = 0
    
    for idx, (img_name, roi_list) in enumerate(roi_dict.items(), 1):
        if idx % 10 != 0:
            continue
        # 尝试在目录中查找图片
        file_name = img_name.split('/')[-1]
        name_without_ext = file_name.rsplit('.', 1)[0]
        parts = name_without_ext.split('_')
        target_parts = parts[-4:-1] 
        file_name = "_".join(target_parts) + ".png"

        img_path = os.path.join(screenshot_dir, file_name)
        
        # 如果直接路径不存在，尝试递归搜索
        if not os.path.exists(img_path):
            # 在目录中递归搜索同名文件
            search_results = list(Path(screenshot_dir).rglob(file_name))
            if search_results:
                img_path = str(search_results[0])
                parent_folder = search_results[0].parent.name
            else:
                if idx % 10 == 0 or idx == 1:
                    print(f"[{idx}/{total_images}] 警告: 未找到图片 - {img_path}")
                continue
        
        processed += 1
        
        try:
            # 打开图片
            img = Image.open(img_path)

            w, h = img.size
            print("w,h origin:", w, h)
            new_w = max(1, int(round(w * 0.5)))
            new_h = max(1, int(round(h * 0.5)))
            new_w = max(16, (new_w // 16) * 16)
            new_h = max(16, (new_h // 16) * 16)
            scaled_images=img.resize((new_w, new_h), Image.BILINEAR)
            
            # 每张图片有5个ROI区域
            if not isinstance(roi_list, list):
                print(f"[{idx}/{total_images}] 错误: ROI列表格式错误 - {img_path}")
                continue
            
            if len(roi_list) < 5:
                print(f"[{idx}/{total_images}] 警告: ROI数量不足 (需要5个，实际{len(roi_list)}) - {img_path}")
            
            # 获取图片基础名称和扩展名
            img_base_name = os.path.splitext(file_name)[0]
            img_ext = os.path.splitext(file_name)[1]
            
            # 裁剪并保存每个ROI
            for roi_idx, coords in enumerate(roi_list, 1):
                if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                    print(f"[{idx}/{total_images}] 警告: 坐标格式错误 (ROI {roi_idx}) - {img_path}")
                    # continue
                
                x1, y1, x2, y2 = coords
                
                # 坐标边界检查
                if x1 < 0 or y1 < 0 or x2 > img.width or y2 > img.height or x1 >= x2 or y1 >= y2:
                    # if idx % 20 == 0:
                    print(f"[{idx}/{total_images}] 警告: 坐标超出范围 (ROI {roi_idx}) - {img_path}")
                    # continue
                
                # 裁剪图片
                cropped = scaled_images.crop((x1, y1, x2, y2))
                
                # 生成输出文件名: 原始名-序号.扩展名
                output_name = f"{img_base_name}-{roi_idx}{img_ext}"
                
                out_dir = os.path.join(output_dir, parent_folder)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

                output_path = os.path.join(out_dir, output_name)
                
                # 保存裁剪后的图片
                cropped.save(output_path)
                total_cropped += 1
            
            # 定期打印进度
            if idx % 10 == 0:
                print(f"进度: [{idx}/{total_images}] 已处理 {processed} 张图片，已裁剪 {total_cropped} 个ROI")
        
        except Exception as e:
            if idx % 20 == 0:
                print(f"[{idx}/{total_images}] 处理错误 - {img_path}: {e}")
            continue
    
    print(f"\n完成!")
    print(f"  总图片数: {total_images}")
    print(f"  已处理: {processed}")
    print(f"  总裁剪数: {total_cropped}")
    print(f"  输出目录: {output_dir}")

def main():
    """主入口函数"""
    print("=" * 60)
    print("screenspotpro ROI 图片裁剪工具")
    print("=" * 60)
    
    # 加载ROI结果
    roi_dict = load_roi_results(ROI_RESULTS_JSON)
    if not roi_dict:
        print("Failed to load ROI results!")
        return 1
    
    print(f"成功加载 {len(roi_dict)} 张图片的ROI坐标")
    
    # 查找图片目录
    screenshot_dir = POSSIBLE_SCREENSHOT_DIRS #find_image_directory()
    if not screenshot_dir:
        print("\n错误: 无法找到图片目录!")
        print(f"请检查以下路径是否存在:")
        for dir_path in POSSIBLE_SCREENSHOT_DIRS:
            print(f"  - {dir_path}")
        return 1
    
    print(f"使用图片目录: {screenshot_dir}")
    
    # 执行裁剪
    crop_and_save_rois(roi_dict, screenshot_dir, OUTPUT_DIR)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

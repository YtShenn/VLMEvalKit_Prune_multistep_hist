import os
import json
from glob import glob
from pathlib import Path

from PIL import Image, ImageDraw

def load_results(json_path):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 两种 evaluate 里 `result` 的结构都类似
    return data

def find_vis_images(vis_dir, img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    # 你现在的 vis 文件名里包含原图 basename，比如 ...__{base}_keep..._drop....
    # pattern = os.path.join(vis_dir, f"*{base}*.png")
    # return glob(pattern)
    pattern = os.path.join(vis_dir, "**", f"*{base}*.png")
    return glob(pattern, recursive=True)

def draw_gt_and_pred(img, parsed_bbox, pred_point, correct: bool):
    w, h = img.size
    draw = ImageDraw.Draw(img, mode="RGBA")

    # bbox / 点都是 0~1 归一化坐标（参考 screenspot_pro.evaluate_point）
    x1 = parsed_bbox[0] * w
    y1 = parsed_bbox[1] * h
    x2 = parsed_bbox[2] * w
    y2 = parsed_bbox[3] * h

    # GT: 绿色框
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)

    if pred_point is not None:
        px = pred_point[0] * w
        py = pred_point[1] * h
        r = max(3, int(0.005 * min(w, h)))  # 小圆点 / 小框半径
        # 预测点：红色小框
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(0,0,255, 255), width=3)

    # 角落写上正确/错误
    text = "CORRECT" if correct else "WRONG"
    color = (0, 255, 0, 255) if correct else (255, 0, 0, 255)
    draw.text((10, 10), text, fill=color)

def main(
    vis_dir: str,
    # failures_json: str,
    # success_json: str,
    res_json: str,
    out_dir: str | None = None,
):
    os.makedirs(out_dir or vis_dir, exist_ok=True)

    results = load_results(res_json)# + load_results(failures_json)
    print(f"loaded {len(results)} records")

    for rec in results:
        img_path = rec["img_path"]
        parsed_bbox = rec["parsed_bbox"]   # 0~1 归一化
        pred = rec.get("pred", None)      # 0~1 归一化 or None
        print("pred: ",pred)
        match = rec.get("match", False)

        vis_paths = find_vis_images(vis_dir, img_path)
        if not vis_paths:
            continue

        for vis_path in vis_paths:
            img = Image.open(vis_path).convert("RGB")
            draw_gt_and_pred(img, parsed_bbox, pred, bool(match))

            # 在文件名后缀加 _correct/_wrong
            stem = Path(vis_path).stem
            suffix = "_correct" if match else "_wrong"
            new_name = stem + suffix + Path(vis_path).suffix

            save_dir = out_dir or vis_dir
            save_path = os.path.join(save_dir, new_name)
            img.save(save_path)
            print("saved:", save_path)

if __name__ == "__main__":
    class_list= ['OS','Creative','CAD','Development','Office','Scientific']
    fail_list = ['/home/ytshen/storage_net2/VLMEvalKit_Prune_my/qwen3vl_after_batch_0.25/screenspot_pro_failure_cases_'+cls+'.json' for cls in class_list]
    succ_list = ['/home/ytshen/storage_net2/VLMEvalKit_Prune_my/qwen3vl_after_batch_0.25/screenspot_pro_success_cases_'+cls+'.json' for cls in class_list]
    
    VIS_DIR = "roi_cropped_images_scale_onimg_0.25"
    # VIS_DIR = '/mnt/storage2/users/ytshen_data/LMUData/images/'
    # FAIL_JSON = os.environ.get("FAILURE_CASES_PATH", "/path/to/failures.json")
    # SUCC_JSON = os.environ.get("SUCCESSFUL_CASES_PATH", "/path/to/success.json")
    # VIS_BBOX_DIR = os.environ.get('VIS_BBOX_DIR', None)
    OUT_DIR = f"/home/ytshen/storage_net2/VLMEvalKit_Prune_my/visualize_pred_bbox_0.25_num=5"  # 或者单独指定一个输出目录

    for i,j in zip(fail_list,class_list):
        FAIL_JSON = i
        out_dir = os.path.join(OUT_DIR, j,"failures")
        main(VIS_DIR, FAIL_JSON, out_dir)

    for i,j in zip(succ_list,class_list):
        FAIL_JSON = i
        out_dir = os.path.join(OUT_DIR, j,"success")
        main(VIS_DIR, FAIL_JSON, out_dir)
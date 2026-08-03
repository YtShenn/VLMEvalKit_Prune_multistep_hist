import json
with open('outputs_gt_50%_nrange20/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_ScreenSpot_Pro_score.json', 'r', encoding='utf-8') as f:
# with open('outputs_50%_textpos/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct_ScreenSpot_Pro_score.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
l=[]
for k in data:
    if 'accuracy' in k or 'Accuracy' in k:
        print(round(data[k],2))
        l.append(round(data[k],2))
print(round(sum(l) / len(l),2))
import csv
from score_single import calculate_metrics

csv_file = "/home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50/checkpoint-138/caption/ours_caption.csv"

# 创建两个字典
caption_token_dict = {}
caption_target_token_dict = {}

with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    
    for i, row in enumerate(reader, start=1):  # 行号从 1 开始
        if len(row) >= 2:  # 确保至少有两列
            caption_target_token_dict[i] = [row[1]]   # 第二列
            caption_token_dict[i] = [row[0]]          # 第一列


# import pdb
# pdb.set_trace()
metric = calculate_metrics(caption_token_dict,caption_target_token_dict)

# 打印前 5 行数据
print("Caption Token Dict (前5行):", dict(list(caption_token_dict.items())[:5]))
print("Caption Target Token Dict (前5行):", dict(list(caption_target_token_dict.items())[:5]))
print("llm_evaluate_metric",metric)

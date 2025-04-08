import os
import re
import math
import json
import argparse
import warnings
import traceback

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append('./')
# from videollama2 import model_init, x_infer
from videollama2.constants import NUM_FRAMES


import random
# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


import copy
from functools import partial

import torch

from videollama2.model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM, Videollama2MixtralForCausalLM
from videollama2.model.builder import load_pretrained_model
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX,IGNORE_INDEX

from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image

from score_single import calculate_metrics

from data.ego4d_data import find_mp4_files,get_annos,preprocess_llama_2_ego4d,ego_video_name_2_video_path
from videollama2 import conversation as conversation_lib

from dataclasses import dataclass, field
import transformers


@dataclass
class DataCollatorForego4dltaDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        # print(instances,55555555555555555555555555555555555555555)
        # import pdb
        # pdb.set_trace()
        batch = dict()
        instance = instances[0]
        batch["timestamp"] = instance["timestamp"]
        batch["labels"] = instance["labels"]
        batch["input_ids"] = instance["input_ids"]
        batch["caption_info"] = instance["caption_info"]
        batch["video_path"] = instance["video_path"]
        batch["images"] = [instance["video"],["video"]]
        batch["attention_mask"] = None
        batch["data_type"] = instance["data_type"]
        batch["model_type"] = instance["model_type"]
        batch["verb_noun"] = instance["verb_noun"]
        return batch



def model_init(model_path,model_base = None,model_name="VideoLLaMA2-7B"):
    # model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    # model_name = get_model_name_from_path(model_path) if model_name is None else  model_name
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
    if tokenizer.unk_token is not None: 
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES

    if 'vicuna' in model_name.lower():
        # vicuna
        version = 'v1'
    elif 'qwen' in model_name.lower():
        # qwen1.5/qwen2
        version = 'qwen'
    else:
        # mistral/mixtral/llama2
        version = 'llama_2'

    return model, partial(process_video, aspect_ratio=None, processor=processor, num_frames=num_frames), tokenizer, version



from data.datasets import LazySupervisedDataset,DataArguments
def build_score_eval(args,tokenizer):
    dataset = LazySupervisedDataset(tokenizer = tokenizer, data_args=args,data_path=None )
    data_collator = DataCollatorForego4dltaDataset(tokenizer=tokenizer)
    # import pdb
    # pdb.set_trace()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)

    # return dict(train_dataset=train_dataset,
    #             eval_dataset=None,
    #             data_collator=data_collator)
    return dataloader,dataset



def is_dataset_caption(timestamp_id,target_timestamp_list,tolerance=5):
    target_ranges = [(t - tolerance, t + tolerance) for t in target_timestamp_list]

    # 统计 TP 和 FP
    tp = 0
    matched_predicted = set()  # 记录已匹配的预测帧
    for i, (start, end) in enumerate(target_ranges):
        if start <= timestamp_id <= end:
            return True,i
    return False, None


def ceil_time_by_fps(time: float, fps: int, min_time: float):
    return max(math.ceil(time * fps) / fps, min_time)


def relaxed_correct(eos_labels, pred_labels, N):
    """
    计算每个位置是否在前后 N 个范围内匹配。
    """
    matches = torch.zeros_like(eos_labels, dtype=torch.bool)
    for i in range(len(eos_labels)):
        start_idx = max(0, i - N)
        end_idx = min(len(eos_labels), i + N + 1)
        if eos_labels[i] in pred_labels[start_idx:end_idx]:
            matches[i] = True
    return matches


import csv
def run_inference_timediff_fluency_ppl_metric(args):
    # 计算videollm-online的指标

    # timediff（eos的出错数量）: 计算整个video的eos数量- 该位置误识别的
    # turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id #
    #frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1

    #llm_ppl = ppl 

    #fluency = correct_eos_token_num + correct_caption_token_num
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)
    model.requires_grad_(False)
    vision_tower = model.get_vision_tower()
    args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor
    model = model.to(torch.bfloat16)

    if args.eval_type == "llm":
        # evaluate LLM
        # import pdb
        # pdb.set_trace()
        llm_val_loader,lta_dataset = build_score_eval(args,tokenizer)
        lta_result = {"predictions":[],"label_ids":[]}
        batch_gt_verb_ids = []
        batch_gt_noun_ids = []
        for data_id, inputs in enumerate(tqdm(llm_val_loader)):
            inputs["input_ids"] = inputs["input_ids"].cuda() 
            inputs["llm_eval"] = True
            gt_verb_ids, gt_noun_ids = inputs["verb_noun"]
            batch_gt_verb_ids.append(gt_verb_ids)
            batch_gt_noun_ids.append(gt_noun_ids)
            # import pdb
            # pdb.set_trace()
            evaluation_kwargs = {'max_new_tokens': 512, 'num_beams': 5, 'num_return_sequences': 5, 'do_sample': False, 'use_cache': True, 'temperature': 1.0, 'top_p': 1.0}
            output = model.generate(inputs = inputs["input_ids"],images_or_videos = inputs["images"][0],modal_list = inputs["images"][1], **evaluation_kwargs,pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            # print(output.loss)
            lta_result["predictions"].append(output)
            lta_result["label_ids"].append(data_id)
        batch_gt_verb_ids = np.array(batch_gt_verb_ids)
        batch_gt_noun_ids = np.array(batch_gt_noun_ids)
        result = lta_dataset.compute_metrics(eval_predictions=lta_result,tokenizer= tokenizer,batch_gt_verb_ids=batch_gt_verb_ids, batch_gt_noun_ids= batch_gt_noun_ids)                
        print(result)


    
   


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--model-path', help='', required=True)
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--device", type=str, required=False, default='cuda:0')
    # parser.add_argument("--batch-size", type=int, default=1)
    # parser.add_argument("--num-workers", type=int, default=8)
    # args = parser.parse_args()

    # run_inference(args)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', default="/home/v-dingxin/blob/finetune_videollama2_mamba_A100_batch1_newcode_1110/checkpoint-150")
    parser.add_argument('--caption-path', default="/home/v-dingxin/videollama2_plus-main/videollama2/eval/ours_caption.csv")
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--eval-caption', default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--cur_fps", type=int, default=2)
    parser.add_argument("--data_type", type=str, default="val")
    parser.add_argument("--eval_type", type=str, default="llm")
    # parser.add_argument("--data_type", type=str, default="train")

    # parser.add_argument('--ego4d_dataset', type=bool,default=True)
    # parser.add_argument('--soccer_dataset_train_llm', type=bool,default=True)
    # parser.add_argument('--soccer_dataset_train_cls',type=bool, default=False)
    # parser.add_argument('--soccer_dataset',type=bool, default=False)
    parser.add_argument(
        "--ego4d_dataset",
        action="store_true",
    )
    parser.add_argument(
        "--ego4d_lta_dataset",
        action="store_true",
    )

  
    parser.add_argument(
        "--soccer_dataset_train_llm",
        action="store_true",
    )
  
    parser.add_argument(
        "--soccer_dataset_train_cls",
        action="store_true",
    )
  
    parser.add_argument(
        "--soccer_dataset",
        action="store_true",
    )
  
    args = parser.parse_args()
    run_inference_timediff_fluency_ppl_metric(args)

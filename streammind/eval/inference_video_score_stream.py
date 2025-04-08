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
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX

from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, process_score_video

from score_single import calculate_metrics

def model_init(model_path,model_base = None,model_name="VideoLLaMA2-7B"):
    # model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    # model_name = get_model_name_from_path(model_path) if model_name is None else  model_name
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)

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


def infer(model, video, instruct, tokenizer, do_sample=False, version='llama_2',score_video = None):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        video (torch.Tensor): video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        version (str): conversation template version.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensor = video.half().cuda()
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + '\n' + instruct
    # import pdb
    # pdb.set_trace()
    conv = conv_templates[version].copy()
    conv.append_message(conv.roles[0], instruct)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_index, return_tensors='pt').unsqueeze(0).cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    kwargs = {"score_video":score_video,"tokenizer":tokenizer}
    # import pdb
    # pdb.set_trace()
    with torch.inference_mode():
        outputs = model.stream_generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modals,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        # import pdb
        # pdb.set_trace()
    
    return outputs

def find_video_files(root_path, target_filenames):
    paths = []
    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(root_path):
        # Check if either of the target files is in the current directory
        for target_filename in target_filenames:
            if target_filename in filenames:
                # Append the full path of the found file
                paths.append(os.path.join(dirpath, target_filename))
    return paths

class ScoreDataset(Dataset):

    def __init__(self,):
        print("*****************getting_finetune_score_data******************")
        # target_filenames = ["1_224p.mkv", "2_224p.mkv"]
        target_filenames = ["1_720p.mkv", "2_720p.mkv"]
        # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime/features_video",target_filenames)
        # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime_debug/features_video",target_filenames)
        self.score_video_list = find_video_files("/mnt/input/MatchTime/features_video",target_filenames)
        # import pdb
        # pdb.set_trace()
        self.caption_path_list = []
        self.remove_video_list_id = []
        for video_id, video_path in enumerate(self.score_video_list):
            caption_path = trans_video_2_json(video_path)
            if os.path.exists(caption_path):
                self.caption_path_list.append(caption_path)
            else:
                self.remove_video_list_id.append(video_id) 
        self.score_video_list = [item for idx,item in enumerate(self.score_video_list) if idx not in self.remove_video_list_id]
        # import pdb
        # pdb.set_trace()
        # print(66)
    def __len__(self):
        return len(self.score_video_list)

    def __getitem__(self, idx):
        num_retries = 50
        for _ in range(num_retries):
            try:
                video_path = self.score_video_list[i]
                
            except:
                i = random.randint(0,len(self.score_video_list) -1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # print(video_path, 646446465464654)
        print("*****************score_data_finetune******************")
        # instruct = f'Question: "What is the content of this video?"\nPlease describe the video content in detail based on the provided information.' 
        # instruct = f'Question: "What has happened in the most recent part of the video?"\nPlease describe the video content in detail based on the provided information.' 
        instruct = f'Please describe the video content in detail based on the provided information.' 
        # instruct = f'6666666666666.' 

        return {
            'video_path': video_path,
            'instruct': instruct,
        }


def build_score_eval(args,):
    dataset = ScoreDataset( )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader


def trans_video_2_json(file_paths):
    # Replace 'features_video' with 'dataset/MatchTime/train'
    new_path = file_paths.replace("features_video", "dataset/MatchTime/train")
    # new_path = file_paths.replace("features_video", "dataset/SN-Caption/test")
    # Replace '1_224p.mkv' or '2_224p.mkv' with 'Labels-caption.json'
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
    elif "1_720p.mkv" in new_path:
        new_path = new_path.replace("1_720p.mkv", "Labels-caption.json")
    elif "2_720p.mkv" in new_path:
        new_path = new_path.replace("2_720p.mkv", "Labels-caption.json")
    
    return new_path

def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    # Use regex to find the number before the underscore
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None

def calculate_cls_metrics(target_list, predicted_list, tolerance=5):
    # 创建标准范围
    target_ranges = [(t - tolerance, t + tolerance) for t in target_list]

    # 统计 TP 和 FP
    tp = 0
    matched_predicted = set()  # 记录已匹配的预测帧
    for pred in predicted_list:
        for start, end in target_ranges:
            if start <= pred <= end:
                tp += 1
                matched_predicted.add(pred)
                break  # 一个预测帧只匹配一个范围

    fp = len(predicted_list) - len(matched_predicted)

    # 统计 FN
    matched_target = set()
    for t in target_list:
        for pred in predicted_list:
            if (t - tolerance) <= pred <= (t + tolerance):
                matched_target.add(t)
                break  # 一个目标帧只匹配一个预测帧

    fn = len(target_list) - len(matched_target)

    # 计算 Precision, Recall, F1-Score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def get_index_stream(start_frame,end_frame , vidoe_fps ,cur_fps = 2):
    
    seg_size = int(vidoe_fps/cur_fps)
    return np.arange(start_frame, end_frame, seg_size, dtype=int)

def read_video_stream(video_path,cur_fps):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    # import pdb
    # pdb.set_trace()
    video_fps = float(vr.get_avg_fps())
    # fps = float(vr.get_avg_fps())
    frame_indices = get_index_stream(start_frame=0,end_frame = max_frame, vidoe_fps = video_fps,cur_fps = cur_fps) 
    return frame_indices, vr


def preprocess_caption_only_caption_data(video_data_path):

    caption_data_path = trans_video_2_json(video_data_path)

    with open(caption_data_path, 'r') as file:
        data = json.load(file)

    timestamp_list = []
    caption_list =  []
    half_list =  []

    half_base = extract_video_half(video_data_path)

    for annotation in data.get('annotations', []):
        gameTime, _ = annotation.get("gameTime",'').split(' - ')
        half = int(gameTime.split(' ')[0])
        if half != half_base:
            continue
        minutes, seconds = map(int, _.split(':'))
        timestamp = minutes * 60 + seconds
        caption_list.append(annotation.get('anonymized', ''))
        timestamp_list.append(timestamp)
        half_list.append(half)
    timestamp_list = timestamp_list[::-1] #让时间从小到大排
    caption_list = caption_list[::-1]
    return timestamp_list, caption_list

import cv2
def run_inference_time_metric(args):
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    precision_list = []
    recall_list = []
    f1_list = []

    precision_list_10 = []
    recall_list_10 = []
    f1_list_10 = []

    precision_list_1 = []
    recall_list_1 = []
    f1_list_1 = []

    for i, line in enumerate(tqdm(val_loader)):
        # if i > 1:
        #     continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i < 5:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        video_frame ,vr = read_video_stream(video_path,2)
        for frame_id in video_frame:
            img = Image.fromarray(vr[frame_id].asnumpy())
            images_group = [img]
            # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
            video_frame = processor(images_group,num_frames=len(images_group))
            # import pdb
            # pdb.set_trace()
            # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
            # video_tensor  = torch.randn(84,3,336,336)
            pred = infer(
                video=video_frame,
                instruct=instruct,
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                version=version,
                score_video=True
            )
            if pred is not None and pred != "":
                print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))
                pred_timestamp_list.append(frame_id//25)
            
            # print("pred")
            # print(pred)
        model.frame_feature = None
        model.past_review_caption = None
        # import pdb
        # pdb.set_trace()
        precision, recall, f1 = calculate_cls_metrics(timestamp_list, pred_timestamp_list)
        precision_10, recall_10, f1_10 = calculate_cls_metrics(timestamp_list, pred_timestamp_list,10)
        precision_1, recall_1, f1_1 = calculate_cls_metrics(timestamp_list, pred_timestamp_list,1)

        precision_list.append(precision)
        precision_list_10.append(precision_10)
        precision_list_1.append(precision_1)

        recall_list.append(recall)
        recall_list_10.append(recall_10)
        recall_list_1.append(recall_1)

        f1_list.append(f1)
        f1_list_10.append(f1_10)
        f1_list_1.append(f1_1)

        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
        print(f"Precision_10: {precision_10:.2f}, Recall_10: {recall_10:.2f}, F1-Score_10: {f1_10:.2f}")
        print(f"Precision_1: {precision_1:.2f}, Recall_10: {recall_1:.2f}, F1-Score_10: {f1_1:.2f}")

    print(f"final Precision: {sum(precision_list)/len(precision_list):.2f}, Recall: {sum(recall_list)/len(recall_list):.2f}, F1-Score: {sum(f1_list)/len(f1_list):.2f}")
    print(f"final Precision_10: {sum(precision_list_10)/len(precision_list_10):.2f}, Recall_10: {sum(recall_list_10)/len(recall_list_10):.2f}, F1-Score_10: {sum(f1_list_10)/len(f1_list_10):.2f}")
    print(f"final Precision_10: {sum(precision_list_1)/len(precision_list_1):.2f}, Recall_10: {sum(recall_list_1)/len(recall_list_1):.2f}, F1-Score_10: {sum(f1_list_1)/len(f1_list_1):.2f}")
    #precision:衡量模型预测的帧中有多少是正确的。,recall:衡量模型是否覆盖了大部分标准帧。
    #TP：模型预测的帧落在任意标准帧范围内的数量。FP：模型预测的帧未落在任何标准帧范围内的数量。FN:标准帧范围内没有被模型预测到的帧数量。
    # ans_file.close()

def is_dataset_caption(timestamp_id,target_timestamp_list,tolerance=5):
    target_ranges = [(t - tolerance, t + tolerance) for t in target_timestamp_list]

    # 统计 TP 和 FP
    tp = 0
    matched_predicted = set()  # 记录已匹配的预测帧
    for i, (start, end) in enumerate(target_ranges):
        if start <= timestamp_id <= end:
            return True,i
    return False, None





def run_inference_caption_metric(args):
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    pred_caption_list = {}
    target_caption_list = {}
    caption_id = 0
    for i, line in enumerate(tqdm(val_loader)):
        # continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i > 1:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal/2_720p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        frame_interval = 1 / args.video_fps

        total_video_frame ,vr = read_video_stream(video_path,args.video_fps)
        cur_min = -1
        cur_sec = -1
        for frame_id in total_video_frame:
            if frame_id < 200:
                continue
            if ((frame_id//25//60 == cur_min) and (frame_id//25%60 > cur_sec)) or (frame_id//25//60 > cur_min) : 
                img = Image.fromarray(vr[frame_id].asnumpy())
                images_group = [img]

                # frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # cv2.imshow("video",frame)

                # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
                video_frame = processor(images_group,num_frames=len(images_group))
                # import pdb
                # pdb.set_trace()
                # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
                # video_tensor  = torch.randn(84,3,336,336)
                pred = infer(
                    video=video_frame,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                    version=version,
                    score_video=True
                )
                # import pdb
                # pdb.set_trace()
                # print(pred)
                if pred is not None and pred != "":
                    cur_min = frame_id // 25 // 60
                    cur_sec = frame_id // 25 % 60
                    # print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))
                    isdataset, caption_id =  is_dataset_caption(frame_id//25,timestamp_list,tolerance=5)
                    if isdataset:
                        print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))
                        # pred_timestamp_list.append(frame_id//25)
                        pred_caption_list[caption_id] = [pred]
                        target_caption_list[caption_id] = [caption_list[caption_id]]
                # if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
                #     break
        # cv2.destroyAllWindows()
        model.frame_feature = None
        model.past_review_caption = None

    result = calculate_metrics(pred_caption_list,target_caption_list)
    print(result)
    #precision:衡量模型预测的帧中有多少是正确的。,recall:衡量模型是否覆盖了大部分标准帧。
    #TP：模型预测的帧落在任意标准帧范围内的数量。FP：模型预测的帧未落在任何标准帧范围内的数量。FN:标准帧范围内没有被模型预测到的帧数量。
    # ans_file.close()


def run_inference_timediff_fluency_ppl_metric(args):
    # 计算videollm-online的指标

    # timediff（eos的出错数量）: 计算整个video的eos数量- 该位置误识别的
    # turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id #
    #frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1

    #llm_ppl = ppl 

    #fluency = correct_eos_token_num + correct_caption_token_num
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    pred_caption_list = {}
    target_caption_list = {}
    caption_id = 0
    for i, line in enumerate(tqdm(val_loader)):
        # continue
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        # if i > 1:                                                          
        #     continue
        # video_path = "/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Crystal_Palace_1_-_2_Arsenal/2_720p.mkv"
        timestamp_list, caption_list = preprocess_caption_only_caption_data(video_path)
        pred_timestamp_list = []
        frame_interval = 1 / args.video_fps

        total_video_frame ,vr = read_video_stream(video_path,args.video_fps)
        cur_min = -1
        cur_sec = -1
        for frame_id in total_video_frame:
            if frame_id < 200:
                continue
            if ((frame_id//25//60 == cur_min) and (frame_id//25%60 > cur_sec)) or (frame_id//25//60 > cur_min) : 
                img = Image.fromarray(vr[frame_id].asnumpy())
                images_group = [img]

                # frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                # cv2.imshow("video",frame)

                # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
                video_frame = processor(images_group,num_frames=len(images_group))
                # import pdb
                # pdb.set_trace()
                # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
                # video_tensor  = torch.randn(84,3,336,336)
                pred = infer(
                    video=video_frame,
                    instruct=instruct,
                    model=model,
                    tokenizer=tokenizer,
                    do_sample=False,
                    version=version,
                    score_video=True
                )
                


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
    parser.add_argument('--model-name', default=None)
    parser.add_argument('--model-base', default=None)
    parser.add_argument('--eval-cls', default=None)
    parser.add_argument('--eval-caption', default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--video_fps", type=int, default=2)
    args = parser.parse_args()
    if args.eval_cls:
        run_inference_time_metric(args)
    elif args.eval_caption:
        run_inference_caption_metric(args)

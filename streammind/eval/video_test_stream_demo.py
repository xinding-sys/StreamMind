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

from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image

from score_single import calculate_metrics

def model_init(model_path,model_base = None,model_name="VideoLLaMA2-7B"):
    # model_path = "DAMO-NLP-SG/VideoLLaMA2-7B" if model_path is None else model_path
    # model_name = get_model_name_from_path(model_path) if model_name is None else  model_name
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    # import pdb
    # pdb.set_trace()
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


def infer(model, video, instruct, tokenizer, do_sample=False, version='mistral_instruct',score_video = None,prompt=None):
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
    # import pdb
    # pdb.set_trace()
    # 1. vision preprocess (load & transform image or video).
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    version='mistral_instruct'
    conv = conv_templates[version].copy()

    tensor = video.half().cuda()
    modals = ["video"]

    if prompt is None:
        # 2. text preprocess (tag process & generate prompt).
        modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
        instruct = modal_token + '\n' 
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
        outputs,cls_pred = model.stream_generate_demo(
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
    if cls_pred == 1:#说明开始新一轮了
        prompt += " " + outputs + " </s>[INST] <video>\n [/INST]"
    return outputs,prompt



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


import cv2
import time
def run_inference_time_metric(args):
    import pdb
    pdb.set_trace()
    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)
    video_path = "/home/v-dingxin/test.mp4"
    instruct = "Please describe the video content in detail based on the provided information."
    # if i < 5:                                                          
    #     continue
    # video_path = "/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv"
    video_frame , vr = read_video_stream(video_path, args.cur_fps)
    cur_min = -1
    cur_sec = -1
    prompt = None
    second_time = 0
    mean_time_list = []
    video_ori_fps = 30
    for frame_id in video_frame:
        if ((frame_id // video_ori_fps // 60 == cur_min) and (frame_id // video_ori_fps % 60 > cur_sec)) or (frame_id // video_ori_fps // 60 > cur_min) : 
            img = Image.fromarray(vr[frame_id].asnumpy())
            images_group = [img]
            video_frame = processor(images_group,num_frames=len(images_group))
            # import pdb
            # pdb.set_trace()
            pred,prompt = infer(
                video=video_frame,
                instruct=instruct,
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                version=version,
                score_video=True,
                prompt = prompt,
            )
            
            if pred is not None and pred != "":#模型认为不是eos的
                    print("The content of the video until {}:{}  is: {}:".format(frame_id//25//60,frame_id//25%60,pred))

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
    parser.add_argument("--cur_fps", type=int, default=2)
    args = parser.parse_args()

    run_inference_time_metric(args)


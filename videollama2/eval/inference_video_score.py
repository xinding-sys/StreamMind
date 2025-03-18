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

    kwargs = {"score_video":score_video}
    with torch.inference_mode():
        output_ids = model.generate(
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

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

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

        # self.processor = processor

        print("*****************getting_finetune_score_data******************")
        target_filenames = ["1_224p.mkv", "2_224p.mkv"]
        # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime/features_video",target_filenames)
        self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime_debug/features_video",target_filenames)

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





def process_score_video(timestamp,processor,video_path,cur_fps = 2):
    def get_index(end_frame, video_fps, max_frame, cur_fps,first_idx=0,start_frame = 0):
        seg_size = int(video_fps/cur_fps)
        return np.arange(start_frame, end_frame, seg_size, dtype=int)

    def load_adjusted_features(duration, timestamp, window, video_fps=25):
        start_frame = int(max(0, timestamp - window) * video_fps + 1)
        if (timestamp + window) * video_fps + 1 > duration:
            return None , None 
        end_frame = int((timestamp + window) * video_fps + 1)

        return start_frame,end_frame

    decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
    # print(decord_vr)
    duration, video_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    # end_time = timestamp + 20
    start_frame,end_frame = load_adjusted_features(duration,timestamp,window = 20, video_fps = video_fps)
    if end_frame is None :
        return None
    frame_id_list = get_index(start_frame = start_frame, end_frame = end_frame, video_fps = video_fps,max_frame = duration,  cur_fps=cur_fps )
    images_group = list()

    for frame_index in frame_id_list:
        img = Image.fromarray(decord_vr[frame_index].asnumpy())
        images_group.append(img)
    # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
    torch_imgs = processor(images_group,num_frames=len(images_group))
    # import pdb
    # pdb.set_trace()
    return torch_imgs 


def run_inference(args):
    def get_index(end_time, fps, max_frame, cur_fps,first_idx=0,start_time = 0):
        if end_time:
            start, end = start_time, end_time
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = int(fps/cur_fps)
        return np.arange(start_idx, end_idx, seg_size, dtype=int)


    def read_video(video_path,processor,cur_fps,end_time=None,start_time = None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = get_index(end_time, fps, max_frame,cur_fps, first_idx=0,start_time=start_time) 
        # print(frame_indices)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        # images_group = [expand2square(img, tuple(int(x*255) for x in self.processor.image_mean)) for img in images_group]
        torch_imgs = processor(images_group,num_frames=len(images_group))
        # import pdb
        # pdb.set_trace()
        return torch_imgs

    model, processor, tokenizer, version = model_init(args.model_path, model_base = args.model_base, model_name = args.model_name)

    val_loader = build_score_eval(args)

    # NOTE: only support batch size 1 for now
    for i, line in enumerate(tqdm(val_loader)):
        video_path = line['video_path'][0]
        instruct = line['instruct'][0]
        import pdb
        pdb.set_trace()
        # if i < 5:                                                          
        #     continue
        video_path = "/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/1_224p.mkv"
        print(video_path)

        for end in range(40,7000,20):
            start = end-40
            video_tensor = read_video(video_path,processor,2,end,start )
            # video_tensor = process_score_video(1112,processor,"/home/v-dingxin/blob/MatchTime_debug/features_video/england_epl_2014-2015/2015-04-11_-_19-30_Burnley_0_-_1_Arsenal/2_224p.mkv",2 )
            # video_tensor  = torch.randn(84,3,336,336)
            pred = infer(
                video=video_tensor,
                instruct=instruct,
                model=model,
                tokenizer=tokenizer,
                do_sample=False,
                version=version,
                score_video=True
            )
            start_min = start//60
            end_min = (end)//60
            start_sec = start%60
            end_sec = (end)%60
            # import pdb
            # pdb.set_trace()
            print("The content of the video from {}:{} to {}:{} seconds is: {}:".format(start_min,start_sec,end_min,end_sec,pred))
            # print("pred")
            # print(pred)

    ans_file.close()

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
    # parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    # parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    run_inference(args)

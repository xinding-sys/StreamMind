import os
import json
import math
import argparse
import warnings
from tqdm import tqdm

import sys
sys.path.append('./')
from videollama2 import model_init, x_infer
import torch

import copy
from functools import partial
from videollama2.model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM, Videollama2MixtralForCausalLM
from videollama2.model.builder import load_pretrained_model
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.mm_utils import process_video, tokenizer_MMODAL_token, get_model_name_from_path, KeywordsStoppingCriteria
from videollama2.constants import NUM_FRAMES, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX




# NOTE: Ignore TypedStorage warning, which refers to this link~(https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run_inference(args):
    # Initialize the model
    # import pdb
    # pdb.set_trace()
    # print(args.model_path,6465465464646)
    model, processor, tokenizer, version = model_init(args.model_path)
    print(model)
    gt_questions = json.load(open(args.question_file, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.answer_file, "r"))
    gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answer_file = os.path.join(args.output_file)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for idx, sample in enumerate(tqdm(gt_questions)):
        # print(idx, sample)
        video_name = sample['video_name']
        question = sample['question']
        qid = sample['question_id']
        answer = gt_answers[idx]['answer']
        video_path = None
        # Load the video file
        for fmt in video_formats:
            temp_path = os.path.join(args.video_folder, f"v_{video_name}{fmt}")
            # print(temp_path,13212131321312)
            if os.path.exists(temp_path):
                video_path = temp_path
                break
            # BUG: compatibility for MSVD, MSRVTT, TGIF
            temp_path = os.path.join(args.video_folder, f"{video_name}{fmt}")
            
            if os.path.exists(temp_path):
                video_path = temp_path
                break
        if video_path is None:
            continue
        # question = question + '\n' + 'Answer the question using a single word or a short phrase with multiple words.'
        # print(video_path,66666666666666)
        video_tensor = processor(video_path)
        output = infer(
            video = video_tensor,
            instruct=question, 
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            version=version,
        )

        sample_set = {'id': qid, 'question': question, 'answer': answer, 'pred': output}
        ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()


def infer(model, video, instruct, tokenizer, do_sample=False, version='llama_2'):
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
    # tensor = [video.half().cuda()]
    tensor = [video.half().cuda()]
    modals = ["video"]

    # 2. text preprocess (tag process & generate prompt).
    modal_token = DEFAULT_MMODAL_TOKEN['VIDEO']
    modal_index = MMODAL_TOKEN_INDEX["VIDEO"]
    instruct = modal_token + '\n' + instruct

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
    # print(prompt,465465464566464)
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
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', help='', required=True)
    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--answer-file', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    args = parser.parse_args()

    run_inference(args)

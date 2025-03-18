# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

# torch-related packages
# NOTE: torch must be imported before transformers. Otherwise, `Segmentation fault (core dumped)` will occur.
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor

# import cv2
import decord
import imageio
import numpy as np
import transformers
from PIL import Image
from decord import VideoReader, cpu
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

sys.path.append('./')
from videollama2 import conversation as conversation_lib
from videollama2.model import *
from videollama2.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, process_score_video
from videollama2.videollama2_trainer_score import (
    VideoLLaMA2Trainer,
    maybe_zero_3, get_mm_adapter_state_maybe_zero_3,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)
import re

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    score_dataset_train_cls: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Other Arguments
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    pretrain_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "To train from previously trained checkpoints. E.g, further fine-tuning based on the finetuned version of the whole model."})


@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # image_folder: Optional[str] = field(default=None)
    # video_folder: Optional[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'
    score_dataset: bool = False
    score_dataset_train_llm: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # NOTE: scan token of each modal and move them to the beginning of the sentence. 
            for DEFAULT_TOKEN in DEFAULT_MMODAL_TOKEN.values():
                MODAL_TYPE = None
                if DEFAULT_TOKEN in sentence['value']:
                    MODAL_TYPE = DEFAULT_TOKEN[1:-1]
                    sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_TOKEN, f'<{MODAL_TYPE.capitalize()}>' + DEFAULT_TOKEN + f'</{MODAL_TYPE.capitalize()}>')
                replace_token = DEFAULT_TOKEN
                if data_args.mm_use_im_start_end and MODAL_TYPE is not None:
                    replace_token = DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()] + replace_token + DEFAULT_MMODAL_START_TOKEN[MODAL_TYPE.upper()]
                sentence["value"] = sentence["value"].replace(DEFAULT_TOKEN, replace_token)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if len(MODAL_list) > 0:
        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print(input_ids,66666666666666666666666666666666666666)
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer))
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
                instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    print(input_ids,999999999999999999999999999999999999999)
    print(targets,888888888888888888888888888)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )





















def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list = [],
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    assert len(sources) == len(MODAL_list)
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        # source is the conversations in the input data
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if len(MODAL_list) > 0:
        # input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    #for conversation, target in zip(conversations, targets):
    for idx, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if len(MODAL_list) > 0:
                # round_len = len(tokenizer_image_token(rou, tokenizer)) 
                # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                # fix the issue of tokenization mismatch
                round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
                instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list=[]
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    DEFAULT_TOKEN = DEFAULT_MMODAL_TOKEN[MODAL_list[0]]
    for source in sources:
        assert len(source) == 2
        source[0]['value'] = DEFAULT_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]], return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_MMODAL_token(source[0]['value'], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]]))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    MODAL_list: list = []
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, MODAL_list)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, MODAL_list)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts, token_index):
        return [len(tokenizer_MMODAL_token(prompt, tokenizer, token_index)) for prompt in prompts]

    if len(MODAL_list) > 0:
        input_ids = [tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for idx, (target, source) in enumerate(zip(targets, sources)):
        if len(MODAL_list) > 0:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source], MODAL_list[idx])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

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

def trans_video_2_json(file_paths):
    # Replace 'features_video' with 'dataset/MatchTime/train'
    new_path = file_paths.replace("features_video", "dataset/MatchTime/train")
        
    # Replace '1_224p.mkv' or '2_224p.mkv' with 'Labels-caption.json'
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
    
    return new_path

def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    # Use regex to find the number before the underscore
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None
    
def parse_labels_caption(caption_data_path,video_data_path,tokenizer):
    """
    Parses a Labels-caption.json file and extracts the required data.
    Parameters:
        file_path (str): The path to the Labels-caption.json file.
        league (str): The league name.
        game (str): The game name.
    Returns:
        list: A list of tuples containing (half, timestamp, type, anonymized, league, game).
    """
    with open(caption_data_path, 'r') as file:
        data = json.load(file)

    label_result = []
    anonymized_result = []
    anonymizeds = []
    timestamp_result = []
    half_result = []
    half_base = extract_video_half(video_data_path)


    for annotation in data.get('annotations', []):
        gameTime, _ = annotation.get("gameTime",'').split(' - ')
        half = int(gameTime.split(' ')[0])
        if half != half_base:
            continue
        minutes, seconds = map(int, _.split(':'))
        timestamp = minutes * 60 + seconds
        label_result.append(annotation.get('label', ''))
        anonymizeds.append(annotation.get('anonymized', ''))
        timestamp_result.append(timestamp)
        half_result.append(half)

    for anony in anonymizeds: 
        anonymized_tokens = tokenizer(
            anony,
            return_tensors = "pt",
            max_length=tokenizer.model_max_length,
            truncation=True
            ).input_ids
        anonymized_result.append(anonymized_tokens)#这个里有很多的caption的token

    input_ids_template = torch.cat((torch.tensor([tokenizer.convert_tokens_to_ids("<|begin_of_text|>")]),
                       torch.tensor([MMODAL_TOKEN_INDEX["VIDEO"]]),
                       torch.tensor([tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) # add end token

    targets_template = torch.cat((torch.tensor([IGNORE_INDEX]),
                       torch.tensor([MMODAL_TOKEN_INDEX["VIDEO"]]),
                       torch.tensor([tokenizer.convert_tokens_to_ids("<|end_of_text|>")]))) # add end token
                       
    return {"labels" : targets_template,
        "video":[],
        "input_ids" : input_ids_template,
        "anonymized" : anonymized_result,#已经token化了
        "caption_info": anonymizeds,
        "timestamp" : timestamp_result,
        "caption_path": caption_data_path,
        "half":half_result}




def video_timestamp_to_video(video_path,timestamp,half,fps=2,input_device=None):
    # if input_device is None:
    input_device = "cpu"
    video_fps = 25
    segment = video_fps//fps
    frame_stamp = 25 * timestamp
    # import pdb
    # pdb.set_trace()
    video_encode_list = []
    video_dir = os.path.dirname(video_path)
    video_encode_feature_dir = video_dir.replace("features_video", "features_video_encode_ddp")
    video_encode_feature_dir = video_encode_feature_dir.replace("MatchTime_debug", "MatchTime")

    if (frame_stamp//500) == 0:
        video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,0,500))
        if frame_stamp + 100 < 500:
            video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp+100:segment])
        else:
            video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp:segment])
    else:
        for video_encode_id in range(frame_stamp//500):
            video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,video_encode_id*500,(video_encode_id+1)*500))
            if os.path.exists(video_encode_path):
                frame_id = frame_stamp % 500
                if video_encode_id == (frame_stamp//500) - 1:
                    if frame_stamp % 500 + 100 < 500:
                        video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp % 500+100:segment])
                    else:
                        video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp % 500:segment])
                else:
                    video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,::segment])
            else:
                continue
    torch.cuda.empty_cache()
    return video_encode_list


def preprocess_llama_2_score(
    caption_data,video_data,half,timestamp,tokenizer
) -> Dict:
    # import pdb
    # pdb.set_trace()
    MODAL_list=['VIDEO']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # caption_data = "please let me pass"
    sources = [[{'from': 'human', 'value': '<video>\nPlease describe the video content in detail based on the provided information.'}, {'from': 'gpt', 'value': caption_data}]]
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    

    input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    if input_ids[0][-3] == 2:#这个是对eos输出的处理
        mask = torch.ones(input_ids.size(), dtype=torch.bool)
        mask[0][-2]= False#这个是为了去掉在token时的一个空格
        input_ids = input_ids[mask].unsqueeze(0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        EOS = "</s>"
        for idx, (conversation, target) in enumerate(zip(conversations, targets)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)#按照con.sep2：(eos token:<\s>)来识别有几轮对话
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):#一轮一轮处理
                if rou == "":
                    break
                rou = rou + EOS
                parts = rou.split(sep)#这样就分成了[instruction+question , answer]
                if len(parts) != 2:
                    break
                parts[0] += sep

                if len(MODAL_list) > 0:
                    # round_len = len(tokenizer_image_token(rou, tokenizer))
                    # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                    round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))#这个是整个对话的token长度
                    instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 1#这个是instruction+question的token长度
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."f" (ignored)")
    else:
        # print(caption_data,conversations,input_ids)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        for idx, (conversation, target) in enumerate(zip(conversations, targets)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)#按照con.sep2：(eos token:<\s>)来识别有几轮对话
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):#一轮一轮处理
                if rou == "":
                    break

                parts = rou.split(sep)#这样就分成了[instruction+question , answer]
                if len(parts) != 2:
                    break
                parts[0] += sep

                if len(MODAL_list) > 0:
                    # round_len = len(tokenizer_image_token(rou, tokenizer))
                    # instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                    round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))#这个是整个对话的token长度
                    instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 2#这个是instruction+question的token长度
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )
    # print(input_ids)
    return {"labels" :targets,
        "video":video_data,
        "input_ids" : input_ids,
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "half":half,
        "video_path":video_data,
        "past_review_caption":None}


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 num_workers = 0):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        # self.tokenizer.pad_token_id = 128001
        # self.tokenizer.add_tokens(["[PLAYER]","[TEAM]","[COACH]","[REFEREE]","([TEAM])"], special_tokens=True)

        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.score_dataset = self.data_args.score_dataset
        if self.score_dataset:
            print("*****************getting_finetune_score_data******************")
            target_filenames = ["1_224p.mkv", "2_224p.mkv"]
            # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime_debug/features_video",target_filenames)
            # self.score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime/features_video",target_filenames)
            # import pdb
            # pdb.set_trace()
            self.score_video_list = find_video_files("/mnt/input/MatchTime/features_video",target_filenames)
            # self.score_video_list = find_video_files("/mnt/input/MatchTime_debug/features_video",target_filenames)
            self.caption_path_list = []
            self.remove_video_list_id = []
            for video_id, video_path in enumerate(self.score_video_list):
                caption_path = trans_video_2_json(video_path)
                if os.path.exists(caption_path):
                    self.caption_path_list.append(caption_path)
                else:
                    self.remove_video_list_id.append(video_id)
                    
            self.score_video_list = [item for idx,item in enumerate(self.score_video_list) if idx not in self.remove_video_list_id]

            self.caption_dict = dict()
            self.eos_caption_dict = dict()

            self.timestamp_dict = dict()
            self.eos_timestamp_dict = dict()

            self.start_timestamp_dict = dict()
            self.eos_start_timestamp_dict = dict()

            self.half_dict = dict()
            self.caption_num = 0
            self.caption_num_pervideo = dict()
            for video_path_id,video_path in enumerate(self.score_video_list):
                if self.data_args.score_dataset_train_llm:
                    self.preprocess_caption_only_caption_data(video_path,video_path_id)
                else:
                    self.preprocess_caption(video_path, video_path_id)#遍历caption path里面全部caption,并且根据timestamp找到对应的video frame
            # import pdb
            # pdb.set_trace()
            # cur_video_id = 0
            # cur_caption_id = 2
            # video_path = self.score_video_list[cur_video_id]
            # half = extract_video_half(video_path)
            # timestamp = self.timestamp_dict[cur_video_id][cur_caption_id - 1]
            # caption = "".join(self.caption_dict[cur_video_id][cur_caption_id - 1])
            # print(caption)
            # data_dict = preprocess_llama_2_score(caption_data=caption,video_data= video_path,half= half,timestamp = timestamp, tokenizer=self.tokenizer)
            # print(8888)

    def preprocess_caption(self,video_data_path,video_path_id):
        def generate_random_non_uniform_timestamp(a, b, min_points=1, max_points=10):
            # 随机生成中间点的数量
            num_points = random.randint(min_points, max_points)
            # 在范围 [a, b] 内生成随机值
            random_values = [random.uniform(a, b) for _ in range(num_points)]
            # 排序并合并起点和终点
            result = [a] + sorted(random_values) + [b]
            return result


        caption_data_path = trans_video_2_json(video_data_path)

        with open(caption_data_path, 'r') as file:
            data = json.load(file)

        self.timestamp_dict[video_path_id] = []

        self.start_timestamp_dict[video_path_id] = []

        timestamp_list = []
        self.caption_dict[video_path_id] =  []

        caption_list =  []
        self.half_dict[video_path_id] =  []
        half_list =  []

        if video_path_id == 0:
            self.caption_num_pervideo[video_path_id] = 0
        else:
            self.caption_num_pervideo[video_path_id] = self.caption_num_pervideo[video_path_id-1]

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
            self.caption_num += 1
            self.caption_num_pervideo[video_path_id] += 1
        timestamp_list = timestamp_list[::-1] #让时间从小到大排
        caption_list = caption_list[::-1]

        # import pdb
        # pdb.set_trace()
        #上面是把当前正点的caption都处理了，接下来要在这些时间点中插入eos caption
        for timeid, timestamp in enumerate(timestamp_list):
            # start_timestamp = timestamp
            if timeid == 0:
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(timestamp)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
                continue

            #在当前timestamp前插入eos token
            if (timestamp - timestamp_list[timeid - 1]) < 2:
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(timestamp_list[timeid - 1])
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
            else:
                start_timestamp = timestamp_list[timeid - 1]
                eos_num = random.randint(1, max(1, (timestamp-start_timestamp)//30))
                # eos_num = max(0, (timestamp-start_timestamp-1)//3)
                # eos_num = max(0, (timestamp-start_timestamp-1)//5)
                # print(start_timestamp,timestamp,546546546)
                eos_timestamp = sorted(random.sample(range(start_timestamp+1,timestamp), eos_num)) #随机插入生成eos数据
                eos_caption = ["</s>" for i in range(eos_num)]
                eos_starttime = [start_timestamp for i in range(eos_num)]

                # self.eos_timestamp_dict[video_path_id].extend(eos_timestamp)#插入timestamp前的eos的time
                self.timestamp_dict[video_path_id].extend(eos_timestamp)#插入timestamp前的eos的time
                self.timestamp_dict[video_path_id].append(timestamp)#最后插入当前timestamp

                # self.eos_start_timestamp_dict[video_path_id].extend(eos_starttime)
                self.start_timestamp_dict[video_path_id].extend(eos_starttime)
                self.start_timestamp_dict[video_path_id].append(start_timestamp)

                # self.eos_caption_dict[video_path_id].extend(eos_caption)
                self.caption_dict[video_path_id].extend(eos_caption)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                # print(eos_num,8464654135131)
                self.caption_num += eos_num
                self.caption_num_pervideo[video_path_id] += eos_num
        # import pdb
        # pdb.set_trace()


    def preprocess_caption_only_caption_data(self,video_data_path,video_path_id):

        caption_data_path = trans_video_2_json(video_data_path)

        with open(caption_data_path, 'r') as file:
            data = json.load(file)

        self.timestamp_dict[video_path_id] = []

        self.start_timestamp_dict[video_path_id] = []

        timestamp_list = []
        self.caption_dict[video_path_id] =  []

        caption_list =  []
        self.half_dict[video_path_id] =  []
        half_list =  []

        if video_path_id == 0:
            self.caption_num_pervideo[video_path_id] = 0
        else:
            self.caption_num_pervideo[video_path_id] = self.caption_num_pervideo[video_path_id-1]

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
            self.caption_num += 1
            self.caption_num_pervideo[video_path_id] += 1
        timestamp_list = timestamp_list[::-1] #让时间从小到大排
        caption_list = caption_list[::-1]

        # import pdb
        # pdb.set_trace()
        #上面是把当前正点的caption都处理了，接下来要在这些时间点中插入eos caption
        for timeid, timestamp in enumerate(timestamp_list):
            if timeid == 0:
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(timestamp)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
                continue

            #在当前timestamp前插入eos token
            self.timestamp_dict[video_path_id].append(timestamp)
            self.start_timestamp_dict[video_path_id].append(timestamp_list[timeid - 1])
            self.caption_dict[video_path_id].append(caption_list[timeid])
            self.half_dict[video_path_id].append(half_list[timeid])
            


    def __len__(self):
        if self.score_dataset:
            return self.caption_num
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return self.caption_num


    def process_score_video(self,start_timestamp, end_timestamp, processor, video_path, cur_fps = 2):
        def get_index(end_frame, video_fps, max_frame, cur_fps,first_idx=0,start_frame = 0):
            #通过起始，结束以及目标fps获得framelist
            seg_size = int(video_fps/cur_fps)
            return np.arange(start_frame, end_frame, seg_size, dtype=int)

        def load_adjusted_features(duration, start_timestamp, end_timestamp, video_fps=25):
            #通过timestamp获得起始和结束frame
            # total_frames = int(window * 2 * video_fps)  # Total frames to extract
            start_frame = int(max(0, start_timestamp) * video_fps + 1)
            if end_timestamp * video_fps + 1 > duration  or start_timestamp == end_timestamp:
                return None , None 
            # if end_timestamp * video_fps + 1 > duration:
                # return None , None 
            end_frame = int((end_timestamp ) * video_fps + 1)

            return start_frame,end_frame

        decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
        # print(decord_vr)
        duration, video_fps = len(decord_vr), float(decord_vr.get_avg_fps())
        # end_time = timestamp + 20
        start_frame,end_frame = load_adjusted_features(duration,start_timestamp, end_timestamp, video_fps = video_fps)
        # print(start_frame,end_frame,43546546541651)
        if end_frame is None :
            return None
        frame_id_list = get_index(start_frame = start_frame, end_frame = end_frame, video_fps = video_fps,max_frame = duration,  cur_fps=cur_fps )

        video_data = decord_vr.get_batch(frame_id_list).asnumpy()   
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x * 255) for x in processor.image_mean)) for image in images]
        # print(len(images),start_timestamp,end_timestamp,start_frame,end_frame,duration,video_path)
        if len(images) == 0:
            return None
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        return video 


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.score_dataset:
            # print(i,374579237495723948759739745)
            num_retries = 50
            video_processor = self.data_args.video_processor
            for video_id, value in self.caption_num_pervideo.items():
                if value > i:
                    cur_video_id = video_id
                    if cur_video_id == 0:
                        cur_caption_id = i
                    else:
                        cur_caption_id = i - self.caption_num_pervideo[cur_video_id-1]
                    break

            video_path = self.score_video_list[cur_video_id]
            half = extract_video_half(video_path)
            timestamp = self.timestamp_dict[cur_video_id][cur_caption_id - 1]
            if 25 * timestamp < 200:
                i = random.randint(0,self.caption_num - 1)
                return self.__getitem__(i)

            start_timestamp = self.start_timestamp_dict[cur_video_id][cur_caption_id - 1]
            caption = "".join(self.caption_dict[cur_video_id][cur_caption_id - 1])
            past_review_caption = "".join("".join(self.caption_dict[cur_video_id][:cur_caption_id - 1]).split("</s>"))
            # print(past_review_caption,646546546546465)
            past_review_caption_tokenids = self.tokenizer(
                            past_review_caption,
                            return_tensors="pt",
                            padding="longest",
                            max_length=self.tokenizer.model_max_length,
                            truncation=True,
                            ).input_ids
            # print(caption)
            # print(past_review_caption)

            
            # continue #太短了video不用了
            # print(caption,46546541651)
            data_dict = preprocess_llama_2_score(caption_data=caption,video_data= video_path,half= half,timestamp = timestamp, tokenizer=self.tokenizer)
            data_dict["past_review_caption"] = past_review_caption_tokenids

            video = self.process_score_video(video_path=video_path,start_timestamp =start_timestamp, end_timestamp = timestamp, processor=video_processor,cur_fps=2)

            if video is None:
                i = random.randint(0,self.caption_num - 1)
                return self.__getitem__(i)

            data_dict["video"] = video
            return data_dict



@dataclass
class DataCollatorForScoreDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances,55555555555555555555555555555555555555555)
        batch = dict()
        instance = instances[0]
        batch["timestamp"] = instance["timestamp"]
        batch["labels"] = instance["labels"]
        batch["input_ids"] = instance["input_ids"]
        batch["half"] = instance["half"]
        batch["caption_info"] = instance["caption_info"]
        batch["video_path"] = instance["video_path"]
        batch["images"] = [instance["video"],["video"]]
        batch["attention_mask"] = None
        batch["past_review_caption"] = instance["past_review_caption"]
        return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        Xs, keys = [], []
        for instance in instances:
            for x in DEFAULT_MMODAL_TOKEN.keys():
                x = x.lower()
                if x in instance:
                    Xs.append(instance[x])
                    keys.append(x)
        batch['images'] = [Xs, keys]  # we do not change the key's name.
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_supervised_score_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForScoreDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    set_seed(42)
    # import pdb
    # pdb.set_trace()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:

        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(dict(
    
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))
    if model_args.pretrain_model_name_or_path is not None:
        assert os.path.exists(model_args.pretrain_model_name_or_path)
        pretrain_model_name_or_path = model_args.pretrain_model_name_or_path
    else:
        pretrain_model_name_or_path = model_args.model_name_or_path
    if model_args.vision_tower is not None:
        if 'vicuna' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2LlamaForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'mistral' in model_args.model_name_or_path.lower():
            # import pdb
            # pdb.set_trace()
            # model_args.model_name_or_path = "/home/v-dingxin/blob/VideoLLaMA2-7B"
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MistralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'llama2' in model_args.model_name_or_path.lower():
            # import pdb
            # pdb.set_trace()
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MistralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
            # import pdb
            # pdb.set_trace()
        elif 'mixtral' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MixtralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
            import deepspeed
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        else:
            # config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            # config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=False)
            config._attn_implementation = attn_implementation
            model = Videollama2MistralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args)
    else:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            if model_args.version == "v1":
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            elif model_args.version == "v1_mistral":
                conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]
    # import pdb
    # pdb.set_trace()
    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)#这个是再videollama2中加入clipmodel

        # import pdb
        # pdb.set_trace()
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        
        # import pdb
        # pdb.set_trace()
        if model_args.score_dataset_train_cls:
            model.requires_grad_(False)
            # model.model.layers[0].self_attn.q_proj.weight.requires_grad = True
            for name, p in model.get_model().mm_projector.named_parameters():
                if "cls" in name:
                    p.requires_grad = True
        # import pdb
        # pdb.set_trace()            

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_MM_tokenizer(model_args, tokenizer=tokenizer)

        model.config.num_frames = NUM_FRAMES if data_args.num_frames is None else data_args.num_frames

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    print("Current model:", model)

    data_module = make_supervised_score_data_module(tokenizer=tokenizer, data_args=data_args)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer
    trainer = VideoLLaMA2Trainer(model=model, data_args = data_args, tokenizer=tokenizer, args=training_args, **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

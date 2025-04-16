import os
import  json
from typing import Dict, Optional, Sequence, List
from streammind import conversation as conversation_lib
import torch

from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image


def find_mp4_files(directory):
    mp4_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

def get_annos(split):
    anno_path = os.path.join("dataset/ego4d/v2/annotations", f'refined_narration_stream_{split}.json')
    assert os.path.exists(anno_path)
    narration_streams = json.load(open(anno_path))
    return narration_streams

def ego_video_name_2_video_path(video_name):
    root_path = "dataset/ego4d/v2/full_scale"
    return os.path.join(root_path,video_name + ".mp4")

from streammind.conversation import conv_templates, SeparatorStyle

def preprocess_llama_2_live(
    caption_data,video_data,tokenizer,data_type
) -> Dict:
    MODAL_list=['VIDEO']
    # conv = conversation_lib.default_conversation.copy()
    version='conv_mistral_instruct_LIVE'
    conv = conv_templates[version].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [[]]
    for index, caption in enumerate(caption_data):
        if caption["role"] == "stream":
            sources[0].append({'from': 'human', 'value': '<video>\n'})
        elif caption["role"] == "user":
            sources[0].append({'from': 'human', 'value': '<video>\n ' + caption["content"]})
        elif caption["role"] == "assistant":
            sources[0].append({'from': 'gpt', 'value': caption["content"]})

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2_LIVE

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
            # import pdb
            # pdb.set_trace()
            if len(MODAL_list) > 0:
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
                print(conversations)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : None,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}





def preprocess_llama_2_live_cls(
    caption_data,video_data,tokenizer,data_type
) -> Dict:
    # import pdb
    # pdb.set_trace()
    MODAL_list=['VIDEO']
    # conv = conversation_lib.default_conversation.copy()
    version='conv_mistral_instruct_LIVE'
    conv = conv_templates[version].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [[]]
    task_requirement = caption_data[0]["content"]
    sources[0].append({'from': 'human', 'value': task_requirement + '<video>\n'})
    sources[0].append({'from': 'gpt', 'value': "</silence>"})


    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[i]], return_tensors='pt') for i, prompt in enumerate(conversations)], dim=0)
    

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2_LIVE

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
                round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
                instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

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
    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : None,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"cls"}

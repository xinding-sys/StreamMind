import re,editdistance
import numpy as np 



import os
import  json
from typing import Dict, Optional, Sequence, List
from streammind import conversation as conversation_lib
import torch

from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image



def get_no_overlap_word(row):
    replace_dict = {
        'pot_(planter)': 'flowerpot',
        'bat_(sports)': 'sport bat',
        'bat_(tool)': 'bat',
        'nut_(food)': 'nuts',
        'nut_(tool)': 'nut',
        'chip_(food)': 'snack',
        "chip_(wood'_metal),": 'chips',
        'chip_(wood,_metal)': 'chip'
    }
    return replace_dict.get(row, split_row_to_words(row)[0])

def split_row_to_words(row):
    if '(' in row:
        words = [re.sub(r'_$', '', row.split('(')[0]).replace('_', ' ')]
        strings = re.sub(r'[)]', '', row.split('(')[1]).split(',')
        strings = [s.lstrip('_').replace('_', ' ') for s in strings]
        words.extend(s for string in strings for s in string.split('/'))
        return words
    else:
        return [row.replace('_', ' ')]

def round_time_by_fps(time: float, fps: int, min_time: float):
    return max(round(time * fps) / fps, min_time)


def get_user_message(num_frames,num_future_actions):
    return {
        "role": "user",
        "content": f"After {num_frames} video frames, anticipate the next {num_future_actions} actions. "
                   f"Format your answer concisely, listing each action on a new line with a number prefix. "
                   f"No extra text output."
    }


def edit_distance(preds: np.ndarray, labels: np.ndarray):
    N, K, Z = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, k, :], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)

def AUED(preds, labels):
    num_future_actions = 20
    ED = np.vstack(
        [edit_distance(preds[:, :, :z], labels[:, :z]) for z in range(1, num_future_actions + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (num_future_actions - 1)
    return AUED.item()


def preprocess_llama_2_ego4d_lta(
    data,tokenizer,data_type
) -> Dict:

    MODAL_list=['VIDEO']
    conv = conversation_lib.default_conversation.copy()
    conv.sep_style = "lta"
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversation_info = data["conversation"]
    sources = [[]]
    sources[0].append({'from': 'human', 'value': conversation_info[0]["content"] + ' <video>\n'})
    sources[0].append({'from': 'gpt', 'value': conversation_info[2]["content"]})

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
    

    targets = input_ids.clone()
    conv.sep_style = conversation_lib.SeparatorStyle.LLAMA_2

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

    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : None,
        "caption_info": conversations,
        "video_path": data["video_path"],
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}



def preprocess_llama_2_ego4d_lta_val_generate(
    data,tokenizer,data_type
) -> Dict:
    MODAL_list=['VIDEO']
    conv = conversation_lib.default_conversation.copy()
    conv.sep_style = "lta"
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversation_info = data["conversation"]
    sources = [[]]
    sources[0].append({'from': 'human', 'value': conversation_info[0]["content"] + ' <video>\n'})

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

    return {"verb_noun":[data['verb_labels'],data['noun_labels']],
        "labels" :None,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : None,
        "caption_info": conversations,
        "video_path": data["video_path"],
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}


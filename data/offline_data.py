import os
import  json
from typing import Dict, Optional, Sequence, List
from streammind import conversation as conversation_lib
import torch

from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image


def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
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
    tokenizer,
    data_type,
    MODAL_list = []
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    if len(MODAL_list) > 0:
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
    if "IMAGE" in MODAL_list:
        data_keys = "image"
    else:
        data_keys = "video"
    return {"labels" :targets,
        data_keys:None,
        "input_ids" : input_ids,
        "timestamp" : None,
        "caption_info":None,
        "video_path":None,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}


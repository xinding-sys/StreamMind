import os 
import re
from typing import Dict, Optional, Sequence, List
from streammind import conversation as conversation_lib
import torch


from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image


def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None


def trans_video_2_json(file_paths,data_type):
    new_path = file_paths.replace("features_video", "dataset/MatchTime/" + data_type)
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
    return new_path


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




def preprocess_llama_2_soccer_cls(
    caption_data,video_data,timestamp,tokenizer,data_type
) -> Dict:
    return {"labels" :None,
        "video":None,
        "input_ids" : torch.tensor([0]),
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"cls"}

def preprocess_llama_2_soccer(
    caption_data,video_data,timestamp,tokenizer,data_type
) -> Dict:
    # import pdb
    # pdb.set_trace()
    MODAL_list=['VIDEO']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources = [[]]
    for caption in caption_data:
        sources[0].append({'from': 'human', 'value': '<video>\n'})
        sources[0].append({'from': 'gpt', 'value': caption})

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

    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}

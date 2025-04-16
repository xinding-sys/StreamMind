import os
import  json
from typing import Dict, Optional, Sequence, List
from streammind import conversation as conversation_lib
import torch

from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image


def find_mp4_files(directory):
    """
    查找目录下所有以 .mp4 结尾的文件，并返回文件路径列表。

    :param directory: 要搜索的目录路径
    :return: 包含所有 .mp4 文件路径的列表
    """
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


def preprocess_llama_2_ego4d(
    caption_data,video_data,timestamp,tokenizer,data_type
) -> Dict:
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

    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}


def process_soccer_video_only_idlist(start_timestamp, end_timestamp, duration, video_fps, cur_fps = 2):
    def get_index(end_frame, video_fps, max_frame, cur_fps,first_idx=0,start_frame = 0):
        seg_size = int(video_fps/cur_fps)
        return np.arange(start_frame, end_frame, seg_size, dtype=int)

    def load_adjusted_features(duration, start_timestamp, end_timestamp, video_fps=25):
        # total_frames = int(window * 2 * video_fps)  # Total frames to extract
        start_frame = int(max(0, start_timestamp) * video_fps-1)
        if end_timestamp * video_fps + 1 > duration  or start_timestamp == end_timestamp:
            return None , None 
        # if end_timestamp * video_fps + 1 > duration:
            # return None , None 
        end_frame = int((end_timestamp ) * video_fps + 1)

        return start_frame,end_frame
    
    start_frame,end_frame = load_adjusted_features(duration,start_timestamp, end_timestamp, video_fps = video_fps)
    if end_frame is None :
        return None
    frame_id_list = get_index(start_frame = start_frame, end_frame = end_frame, video_fps = video_fps,max_frame = duration,  cur_fps=cur_fps )

    return frame_id_list


def preprocess_llama_2_ego4d_cls(
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

def preprocess_llama_2_ego4d_eos(
    caption_data,video_data,timestamp,tokenizer
) -> Dict:
    # import pdb
    # pdb.set_trace()
    MODAL_list=['VIDEO']
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # caption_data = "please let me pass"
    sources = [[{'from': 'human', 'value':'<video>\n'}, {'from': 'gpt', 'value': caption_data}]]
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
    if input_ids[0][-3] == 2:
        mask = torch.ones(input_ids.size(), dtype=torch.bool)
        mask[0][-2]= False
        input_ids = input_ids[mask].unsqueeze(0)

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

        # Mask targets
        sep = "[/INST] "
        EOS = "</s>"
        for idx, (conversation, target) in enumerate(zip(conversations, targets)):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                rou = rou + EOS
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if len(MODAL_list) > 0:
                    round_len = len(tokenizer_MMODAL_token(rou, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]]))
                    instruction_len = len(tokenizer_MMODAL_token(parts[0], tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[idx]])) - 1
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
    # print(input_ids)
    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None}
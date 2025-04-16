import os,collections,itertools,Levenshtein
import sys
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, ToTensor

import decord
import imageio
import numpy as np
import transformers
from PIL import Image
from decord import VideoReader, cpu
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
sys.path.append('./')
from streammind import conversation as conversation_lib



from streammind.model import *
from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image
from streammind.streammind_trainer_score import (
    StreamMindTrainer,
    maybe_zero_3, get_mm_adapter_state_maybe_zero_3,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)
import re

import math
from data.soccer_data import extract_video_half,trans_video_2_json,find_video_files,preprocess_llama_2_soccer,preprocess_llama_2_soccer_cls
from data.ego4d_data import find_mp4_files,get_annos,preprocess_llama_2_ego4d,ego_video_name_2_video_path,preprocess_llama_2_ego4d_cls
from data.live_data import preprocess_llama_2_live,preprocess_llama_2_live_cls
from data.offline_data import preprocess_multimodal,preprocess_llama_2
from data.ego4d_lta import get_no_overlap_word,round_time_by_fps,get_user_message,preprocess_llama_2_ego4d_lta,AUED,preprocess_llama_2_ego4d_lta_val_generate



@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_folder: Optional[str] = field(default=None)
    # Loading Arguments
    is_multimodal: bool = False
    lazy_preprocess: bool = False
    num_frames: Optional[int] = field(default=None)
    cur_fps: Optional[int] = field(default=2)
    # Preprocess Arguments
    image_aspect_ratio: str = 'square'
    data_type: str = "train"
    soccer_dataset: bool = False
    ego4d_dataset:bool = False
    offline_dataset:bool = False
    live_dataset:bool = False
    ego4d_lta_dataset:bool = False
    soccer_dataset_train_llm: bool = False

import tqdm
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""


    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 num_workers = 0):
        super(LazySupervisedDataset, self).__init__()

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.soccer_dataset = self.data_args.soccer_dataset
        self.ego4d_dataset = self.data_args.ego4d_dataset
        self.live_dataset = self.data_args.live_dataset
        self.ego4d_lta_dataset = self.data_args.ego4d_lta_dataset
        self.offline_dataset = self.data_args.offline_dataset
        self.cur_fps = self.data_args.cur_fps
        self.data_type = self.data_args.data_type
        if self.soccer_dataset:
            print("*****************getting_finetune_soccer_data******************")
            target_filenames = ["1_224p.mkv", "2_224p.mkv"]
            self.soccer_video_list = find_video_files("data/MatchTime/features_video",target_filenames)
            self.caption_path_list = []
            self.remove_video_list_id = []
            for video_id, video_path in enumerate(self.soccer_video_list):
                caption_path = trans_video_2_json(video_path,self.data_type)
                if os.path.exists(caption_path):
                    self.caption_path_list.append(caption_path)
                else:
                    self.remove_video_list_id.append(video_id)

            self.soccer_video_list = [item for idx,item in enumerate(self.soccer_video_list) if idx not in self.remove_video_list_id]

            self.caption_dict = dict()
            self.eos_caption_dict = dict()

            self.timestamp_dict = dict()
            self.eos_timestamp_dict = dict()

            self.start_timestamp_dict = dict()
            self.eos_start_timestamp_dict = dict()

            self.half_dict = dict()
            self.caption_num = 0
            self.caption_num_pervideo = dict()
            for video_path_id,video_path in enumerate(self.soccer_video_list):
                self.preprocess_caption_only_caption_data_soccer(video_path,video_path_id,self.data_type)

        if self.ego4d_dataset:
            print("*****************getting_finetune_ego4d_data******************")
            self.ego4d_caption_data = get_annos(self.data_type)
            self.ego4d_video_list = list(self.ego4d_caption_data.keys())
            self.remove_video_list_id = []
            for video_name_id, video_name in enumerate(self.ego4d_video_list):
                if len(self.ego4d_caption_data[video_name].keys())==0:
                    self.remove_video_list_id.append(video_name_id)
            # import pdb
            # pdb.set_trace() 
            self.ego4d_video_list = [item for idx,item in enumerate(self.ego4d_video_list) if idx not in self.remove_video_list_id]

            self.caption_dict = dict()
            self.eos_caption_dict = dict()

            self.timestamp_dict = dict()
            self.eos_timestamp_dict = dict()

            self.start_timestamp_dict = dict()
            self.eos_start_timestamp_dict = dict()

            self.half_dict = dict()
            self.caption_num = 0
            self.caption_num_pervideo = dict()
            for video_name_id, video_name in enumerate(self.ego4d_video_list):
                self.preprocess_caption_only_caption_data_ego4d(video_name,video_name_id)

        if self.ego4d_lta_dataset:
            self.root = "/mnt/input/ego4d/v2"
            self.num_future_actions = 20
            self.num_input_actions = 8
            self.num_beams = 5
            taxonomy = json.load(open(os.path.join(self.root, 'annotations', 'fho_lta_taxonomy.json')))
            self.verbs = [get_no_overlap_word(verb) for verb in taxonomy['verbs']]
            self.nouns = [get_no_overlap_word(noun) for noun in taxonomy['nouns']]
            self.action_to_verb_label, self.action_to_noun_label = {}, {}
            action_counter = collections.defaultdict(int)
            for (i, verb), (j, noun) in itertools.product(enumerate(self.verbs), enumerate(self.nouns)):
                action = f'{verb} {noun}'
                self.action_to_verb_label[action] = i
                self.action_to_noun_label[action] = j
                action_counter[action] += 1
            self.most_common_action = max(action_counter, key=action_counter.get)

            anno_path = os.path.join(self.root, 'annotations', f'fho_lta_{self.data_type}.json')
            annos = json.load(open(anno_path))['clips']
            clip2anno = collections.defaultdict(list)
            for anno in annos:
                clip2anno[anno['clip_uid']].append({
                    'video_uid': anno['video_uid'],
                    'start': anno['clip_parent_start_sec'] + anno['action_clip_start_sec'],
                    'end': anno['clip_parent_start_sec'] + anno['action_clip_end_sec'],
                    'action_idx': anno['action_idx'],
                    'verb_label': anno.get('verb_label'), 'noun_label': anno.get('noun_label', None),
                    'clip_idx': anno.get('clip_idx', None),
                    'clip_uid': anno['clip_uid'],
                })
            clip2anno = {
                clip : sorted(anno, key=lambda x:x['action_idx']) \
                for clip, anno in clip2anno.items() \
                if len(anno) >= self.num_future_actions + self.num_input_actions
            }
            self.clip2anno = clip2anno

            # # 3. make flatten annotations
            self.annos = []
            for clip_uid, anno in clip2anno.items():
                for i in range(len(anno) - self.num_future_actions - self.num_input_actions + 1):
                    video_uid = anno[i]['video_uid']
                    j, k = i + self.num_input_actions, i + self.num_future_actions + self.num_input_actions
                    if 'test_unannotated' in self.data_type:
                        verb_labels, noun_labels = None, None
                    else:
                        verb_noun_labels = [(a['verb_label'], a['noun_label']) for a in anno[j:k]]
                        response = self.verb_noun_labels_to_text(verb_noun_labels)
                        verb_labels, noun_labels = zip(*verb_noun_labels)
                    start_time = round_time_by_fps(anno[i]['start'], self.cur_fps, min_time=0)
                    end_time = round_time_by_fps(anno[j-1]['end'], self.cur_fps, min_time=0)
                    start_frame = int(start_time * self.cur_fps)
                    stop_frame = int(end_time * self.cur_fps) + 1
                    conversation = [
                        get_user_message(stop_frame - start_frame,self.num_future_actions),
                        {"role": "stream", 'num_frames': stop_frame - start_frame},
                    ]
                    if self.data_type == "train":
                        conversation[-1]['learn'] = True
                        conversation.append({"role": "assistant", "content": response, 'learn': True})
                    self.annos.append({
                        'conversation': conversation,
                        'load_ranges': [start_time, end_time],
                        'verb_labels': verb_labels,
                        'noun_labels': noun_labels,
                        'clip_uid': clip_uid,
                        "video_path":video_uid,
                        'last_visible_action_idx': anno[j-1]['action_idx'],
                    })
            # self.annos= self.annos[:11]
            self.annos_verb_labels = np.array([anno['verb_labels'] for anno in self.annos])
            self.annos_noun_labels = np.array([anno['noun_labels'] for anno in self.annos])

        if self.live_dataset:
            self.anno_path = '/mnt/input/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json'

            self.embed_dir = "/mnt/input/ego4d/v2/full_scale"
            self.metadata = self.get_metadata()
            self.annos = []
            annos = json.load(open(self.anno_path))
            # index = 0

            for anno in tqdm.tqdm(annos):
                video_uid = anno['video_uid']
                try:
                    duration = self.metadata[video_uid]['duration']
                except:
                    continue
                if not anno['conversation']:
                    continue
                role = anno['conversation'][0]['role']
                time = anno['conversation'][0]['time']
                content = anno['conversation'][0]['content']
                if not (role == 'user' and time > 0 and time <= duration and content):
                    continue
                frame_fps = 30
                # 1. add random frames before the user
                fps_time = self.floor_time_by_fps(time, frame_fps, 0, duration)
                waiting_frames = random.randint(0, min(20, int(fps_time * frame_fps)))
                conversation = []
                if waiting_frames:
                    conversation.append({'role': 'user', 'num_frames': waiting_frames, 'learn': waiting_frames - 1,'content': content, 'time': time, 'fps_time': fps_time})
                else:
                    conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
                start_fps_time = fps_time - (waiting_frames - 1) / frame_fps
                # 2. for loop to add message
                for message in anno['conversation'][1:]:
                    role, content, time = message['role'], message['content'], message['time']
                    if time > duration:
                        break
                    if time < conversation[-1]['time']:
                        break
                    if time == conversation[-1]['time']:
                        if role == 'user':
                            break
                        else:
                            if conversation[-1]['role'] == 'user':
                                conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': conversation[-1]['fps_time'], 'learn': True})
                            else:
                                conversation[-1]['content'] = content
                            continue
                    if role == 'user':
                        fps_time = self.floor_time_by_fps(time, frame_fps, conversation[-1]['fps_time'], duration)
                        if fps_time > duration:
                            break
                        if fps_time > conversation[-1]['fps_time']:
                            # conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                            conversation.append({'role': 'user','num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True, 'content': content, 'time': time, 'fps_time': fps_time})
                        else:
                            conversation.append({'role': 'user', 'content': content, 'time': time, 'fps_time': fps_time})
                    else:
                        fps_time = self.ceil_time_by_fps(time, frame_fps, conversation[-1]['fps_time'])
                        if fps_time > duration:
                            break
                        if fps_time > conversation[-1]['fps_time']:
                            conversation.append({'role': 'stream', 'num_frames': int((fps_time - conversation[-1]['fps_time']) * frame_fps), 'learn': True})
                            conversation.append({'role': 'assistant', 'content': content, 'time': time, 'fps_time': fps_time, 'learn': True})
                if not conversation:
                    continue

                timestamp = []
                fps_time = -10
                for conv_id,conv in enumerate(conversation):
                    if "fps_time" in conv and conv["fps_time"] != fps_time:
                        fps_time = conv["fps_time"]
                        timestamp.append(fps_time)
                if timestamp[0] <= 1:
                    start_time = 0
                else: 
                    start_time = max(0,timestamp[0] - random.randint(1, min(20, int(timestamp[0]))))
                start_timestamp = [start_time]+timestamp[:-1]

                self.annos.append({
                    'conversation': conversation,
                    "path":self.metadata[video_uid]['path'],
                    'load_ranges': range(int(start_fps_time*frame_fps), int(conversation[-1]['fps_time']*frame_fps)+1),
                    "timestamp":timestamp,
                    "start_timestamp":start_timestamp
                })
        if self.offline_dataset:
            self.list_data_dict = json.load(open(data_path, "r"))

        
    def get_metadata(self, ):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            # index = 0
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                try:
                    decord_vr = VideoReader(uri=path, ctx=cpu(0), num_threads=1)
                except:
                    continue
                # print(decord_vr)
                duration, video_fps = len(decord_vr), float(decord_vr.get_avg_fps())
                duration = duration / video_fps
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata


    def ceil_time_by_fps(self,time: float, fps: int, min_time: float):
        return max(math.ceil(time * fps) / fps, min_time)

    def floor_time_by_fps(self,time: float, fps: int, min_time: float, max_time: float):
        return min(max(math.floor(time * fps) / fps, min_time), max_time)

    ###########################ego4d_lta###########################
    def get_labels(self, indices):
        return self.annos_verb_labels[indices], self.annos_noun_labels[indices]

    def verb_noun_labels_to_text(self, verb_noun_labels: list[tuple[str]]):
        return '\n'.join([f'{i+1}. {self.verbs[v].capitalize()} {self.nouns[n]}.' for i, (v, n) in enumerate(verb_noun_labels)])

    def map_action_to_verb_label(self, action: str):
        if action not in self.action_to_verb_label:
            action = min([(Levenshtein.distance(action, key), key) for key in self.action_to_verb_label.keys()])[1]
        return self.action_to_verb_label[action]

    def map_action_to_noun_label(self, action: str):
        if action not in self.action_to_noun_label:
            action = min([(Levenshtein.distance(action, key), key) for key in self.action_to_noun_label.keys()])[1]
        return self.action_to_noun_label[action]

    def text_to_verb_noun_ids(self, text: str, num_actions: int):
        actions = []
        text = text.strip(' \n')
        for line in text.split('\n'):
            match = re.search(r'(?:\d+\.|[^\s]+\s\d+\.)\s*(.*)', line)
            if match:
                actions.append(match.group(1).lower().rstrip('.'))
        verb_noun_ids = [(self.map_action_to_verb_label(action), self.map_action_to_noun_label(action)) for action in actions]
        verb_noun_ids = verb_noun_ids[:num_actions]
        if len(verb_noun_ids) < num_actions:
            if verb_noun_ids:
                verb_noun_ids = verb_noun_ids + [verb_noun_ids[-1]] * (num_actions - len(verb_noun_ids))
            else:
                verb_noun_ids = [(
                    self.map_action_to_verb_label(self.most_common_action),
                    self.map_action_to_noun_label(self.most_common_action)
                )] * num_actions
        return verb_noun_ids

    def compute_metrics(self, eval_predictions, tokenizer, batch_gt_verb_ids, batch_gt_noun_ids, output_dir: str = './'):

        batch_beam_pred_tensor, sample_idxs = eval_predictions["predictions"], eval_predictions["label_ids"]
        batch_beam_pred_verb_ids, batch_beam_pred_noun_ids = [], []
        for beam_pred_tensor in batch_beam_pred_tensor:
            beam_pred_tensor = beam_pred_tensor[beam_pred_tensor != -100].reshape(self.num_beams, -1)
            beam_pred_string = tokenizer.batch_decode(beam_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            beam_verb_noun_ids = np.array([self.text_to_verb_noun_ids(pred_string, self.num_future_actions) for pred_string in beam_pred_string]) # 5 x 20 x 2
            beam_pred_verb_ids, beam_pred_noun_ids = beam_verb_noun_ids[:,:,0], beam_verb_noun_ids[:,:,1]
            batch_beam_pred_verb_ids.append(beam_pred_verb_ids)
            batch_beam_pred_noun_ids.append(beam_pred_noun_ids)
        batch_beam_pred_verb_ids, batch_beam_pred_noun_ids = np.stack(batch_beam_pred_verb_ids), np.stack(batch_beam_pred_noun_ids)
        if 'test_unannotated' not in self.data_type:
            # batch_gt_verb_ids, batch_gt_noun_ids = self.get_labels(sample_idxs)
            return {
                'verb_AUED': AUED(batch_beam_pred_verb_ids, batch_gt_verb_ids),
                'noun_AUED': AUED(batch_beam_pred_noun_ids, batch_gt_noun_ids)
            }
        else:
            predictions = {}
            for beam_pred_verb_ids, beam_pred_noun_ids, sample_idx in zip(batch_beam_pred_verb_ids, batch_beam_pred_noun_ids, sample_idxs):
                clip_uid = self.annos[sample_idx]['clip_uid']
                last_visible_action_idx = self.annos[sample_idx]['last_visible_action_idx']
                key = f'{clip_uid}_{last_visible_action_idx}'
                predictions[key] = dict(verb=beam_pred_verb_ids.tolist(), noun=beam_pred_noun_ids.tolist())
            if torch.cuda.current_device() == 0:
                json.dump(predictions, open(os.path.join(output_dir, f'{self.data_type}_predictions.json'), 'w'))
            return {}
        
    ###########################ego4d_lta###########################




    ###########################ego4d###########################
    def preprocess_caption_only_caption_data_ego4d(self,video_data_name,video_name_id):

        data = next(iter(self.ego4d_caption_data[video_data_name].values()))
    
        self.timestamp_dict[video_name_id] = []

        self.start_timestamp_dict[video_name_id] = []

        timestamp_list = []
        self.caption_dict[video_name_id] =  []

        caption_list =  []

        if video_name_id == 0:
            self.caption_num_pervideo[video_name_id] = 0
        else:
            self.caption_num_pervideo[video_name_id] = self.caption_num_pervideo[video_name_id-1]

        for annotation in data:
            text = annotation.get('text')
            time = annotation.get("time")
            if len(caption_list) > 0:
                if text == caption_list[-1]:
                    continue
            caption_list.append(text)
            timestamp_list.append(time)
            self.caption_num += 1
            self.caption_num_pervideo[video_name_id] += 1

        for timeid, timestamp in enumerate(timestamp_list):
            if timeid == 0:
                timestamp = self.ceil_time_by_fps(timestamp, self.cur_fps, min_time=0)
                start_time = max(0,timestamp - 1 / self.cur_fps)
                self.timestamp_dict[video_name_id].append(timestamp)
                self.start_timestamp_dict[video_name_id].append(start_time)
                self.caption_dict[video_name_id].append(caption_list[timeid])
                continue

            self.timestamp_dict[video_name_id].append(self.ceil_time_by_fps(timestamp, self.cur_fps, min_time=0))
            self.start_timestamp_dict[video_name_id].append(self.ceil_time_by_fps(timestamp_list[timeid - 1], self.cur_fps, min_time=0))
            self.caption_dict[video_name_id].append(caption_list[timeid])

    def preprocess_caption_ego4d(self,video_data_name,video_name_id):
        def generate_random_non_uniform_timestamp(a, b, min_points=1, max_points=10):
            num_points = random.randint(min_points, max_points)
            random_values = [random.uniform(a, b) for _ in range(num_points)]
            result = [a] + sorted(random_values) + [b]
            return result

        data = next(iter(self.ego4d_caption_data[video_data_name].values()))

        self.timestamp_dict[video_name_id] = []

        self.start_timestamp_dict[video_name_id] = []

        timestamp_list = []
        self.caption_dict[video_name_id] =  []

        caption_list =  []

        if video_name_id == 0:
            self.caption_num_pervideo[video_name_id] = 0
        else:
            self.caption_num_pervideo[video_name_id] = self.caption_num_pervideo[video_name_id-1]


        for annotation in data:
            text = annotation.get('text')
            time = annotation.get("time")
            if len(caption_list) > 0:
                if text == caption_list[-1]:
                    continue
            caption_list.append(text)
            timestamp_list.append(time)
            self.caption_num += 1
            self.caption_num_pervideo[video_name_id] += 1


        for timeid, timestamp in enumerate(timestamp_list):
            # start_timestamp = timestamp
            if timeid == 0:
                timestamp = self.ceil_time_by_fps(timestamp, self.cur_fps, min_time=0)
                start_time = max(0,timestamp - 1 / self.cur_fps)
                self.timestamp_dict[video_name_id].append(timestamp)
                self.start_timestamp_dict[video_name_id].append(start_time)
                self.caption_dict[video_name_id].append(caption_list[timeid])
                continue

            start_timestamp = self.ceil_time_by_fps(timestamp_list[timeid - 1], self.cur_fps, min_time=0)
            timestamp = self.ceil_time_by_fps(timestamp, self.cur_fps, min_time=0)
            if (timestamp - start_timestamp) < 2:
                self.timestamp_dict[video_name_id].append(self.ceil_time_by_fps(timestamp, self.cur_fps, min_time=0))
                self.start_timestamp_dict[video_name_id].append(self.ceil_time_by_fps(timestamp_list[timeid - 1], self.cur_fps, min_time=0))
                self.caption_dict[video_name_id].append(caption_list[timeid])
            else:
                eos_num = random.randint(1, max(1, int((timestamp-start_timestamp)//30)))
                eos_timestamp = sorted(random.sample(range(int(start_timestamp+1),int(timestamp)), eos_num)) 
                eos_caption = ["None" for i in range(eos_num)]
                eos_starttime = [start_timestamp for i in range(eos_num)]

                self.timestamp_dict[video_name_id].extend(eos_timestamp)
                self.timestamp_dict[video_name_id].append(timestamp)

                self.start_timestamp_dict[video_name_id].extend(eos_starttime)
                self.start_timestamp_dict[video_name_id].append(start_timestamp)

                self.caption_dict[video_name_id].extend(eos_caption)
                self.caption_dict[video_name_id].append(caption_list[timeid])
                self.caption_num += eos_num
                self.caption_num_pervideo[video_name_id] += eos_num

    ###########################ego4d###########################

    def preprocess_caption_soccer(self,video_data_path,video_path_id):
        def generate_random_non_uniform_timestamp(a, b, min_points=1, max_points=10):
            num_points = random.randint(min_points, max_points)
            random_values = [random.uniform(a, b) for _ in range(num_points)]
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
        timestamp_list = timestamp_list[::-1] 
        caption_list = caption_list[::-1]

        for timeid, timestamp in enumerate(timestamp_list):
            if timeid == 0:
                start_time = min(0,timestamp - 1 / self.cur_fps)
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(start_time)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
                continue

            if (timestamp - timestamp_list[timeid - 1]) < 2:
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(timestamp_list[timeid - 1])
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
            else:
                start_timestamp = timestamp_list[timeid - 1]
                eos_num = int(min(max(0, (timestamp-start_timestamp-1)//3),50))
                eos_timestamp = sorted(random.sample(range(start_timestamp+1,timestamp), eos_num)) 
                eos_caption = ["None" for i in range(eos_num)]
                eos_starttime = [start_timestamp for i in range(eos_num)]

                self.timestamp_dict[video_path_id].extend(eos_timestamp)
                self.timestamp_dict[video_path_id].append(timestamp)

                self.start_timestamp_dict[video_path_id].extend(eos_starttime)
                self.start_timestamp_dict[video_path_id].append(start_timestamp)

                self.caption_dict[video_path_id].extend(eos_caption)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.caption_num += eos_num
                self.caption_num_pervideo[video_path_id] += eos_num


    def preprocess_caption_only_caption_data_soccer(self,video_data_path,video_path_id,data_type):

        caption_data_path = trans_video_2_json(video_data_path,data_type)

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
        timestamp_list = timestamp_list[::-1] 
        caption_list = caption_list[::-1]

        for timeid, timestamp in enumerate(timestamp_list):
            if timeid == 0:
                start_time = min(0,timestamp - 1 / self.cur_fps)
                if start_time < 0:
                    self.caption_num -= 1
                    self.caption_num_pervideo[video_path_id] -= 1
                    continue
                self.timestamp_dict[video_path_id].append(timestamp)
                self.start_timestamp_dict[video_path_id].append(start_time)
                self.caption_dict[video_path_id].append(caption_list[timeid])
                self.half_dict[video_path_id].append(half_list[timeid])
                continue
            start_time = timestamp_list[timeid - 1]
            if start_time == timestamp:
                self.caption_num -= 1
                self.caption_num_pervideo[video_path_id] -= 1
                continue
            self.timestamp_dict[video_path_id].append(timestamp)
            self.start_timestamp_dict[video_path_id].append(start_time)
            self.caption_dict[video_path_id].append(caption_list[timeid])
            self.half_dict[video_path_id].append(half_list[timeid])
            


    def __len__(self):
        if self.soccer_dataset:
            return len(self.soccer_video_list)
        if self.ego4d_dataset:
            return len(self.ego4d_video_list)
        if self.ego4d_lta_dataset:
            return len(self.annos)
        if self.live_dataset:
            return len(self.annos)
        if self.offline_dataset:
            return len(self.list_data_dict)

    @property
    def lengths(self):
        return self.caption_num


    def process_soccer_video(self,start_timestamp, end_timestamp, processor, video_path, cur_fps = 2):
        def get_index(end_frame, video_fps, max_frame, cur_fps,first_idx=0,start_frame = 0):
            seg_size = int(video_fps/cur_fps)
            return np.arange(start_frame, end_frame, seg_size, dtype=int)

        def load_adjusted_features(duration, start_timestamp, end_timestamp, video_fps=25):
            start_frame = int(max(0, start_timestamp) * video_fps-1)
            if end_timestamp * video_fps + 1 > duration  or start_timestamp == end_timestamp:
                return None , None 
            end_frame = int((end_timestamp ) * video_fps + 1)

            return start_frame,end_frame
        try:
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
        except:
            return None
        duration, video_fps = len(decord_vr), float(decord_vr.get_avg_fps())
        start_frame,end_frame = load_adjusted_features(duration,start_timestamp, end_timestamp, video_fps = video_fps)
        if end_frame is None :
            return None
        frame_id_list = get_index(start_frame = start_frame, end_frame = end_frame, video_fps = video_fps,max_frame = duration,  cur_fps=cur_fps )
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()   
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x * 255) for x in processor.image_mean)) for image in images]
        if len(images) == 0:
            return None
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        return video



    def process_soccer_video_only_idlist(self,start_timestamp, end_timestamp, duration, video_fps, cur_fps = 2):
        def get_index(end_frame, video_fps, max_frame, cur_fps,first_idx=0,start_frame = 0):
            seg_size = int(video_fps/cur_fps)
            return np.arange(start_frame, end_frame, seg_size, dtype=int)

        def load_adjusted_features(duration, start_timestamp, end_timestamp, video_fps=25):
            start_frame = int(max(0, start_timestamp) * video_fps-1)
            if end_timestamp * video_fps + 1 > duration  or start_timestamp == end_timestamp:
                return None , None 
            end_frame = int((end_timestamp ) * video_fps + 1)

            return start_frame,end_frame
       
        start_frame,end_frame = load_adjusted_features(duration,start_timestamp, end_timestamp, video_fps = video_fps)
        if end_frame is None :
            return None
        frame_id_list = get_index(start_frame = start_frame, end_frame = end_frame, video_fps = video_fps,max_frame = duration,  cur_fps=cur_fps )

        return frame_id_list
    
    def get_list_video(self,processor,decord_vr,frame_id_list):

        video_data = decord_vr.get_batch(frame_id_list).asnumpy()   
        images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
        images = [expand2square(image, tuple(int(x * 255) for x in processor.image_mean)) for image in images]
        if len(images) == 0:
            return None
        video = processor.preprocess(images, return_tensors='pt')['pixel_values']
        return video 



    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_retries = 50
        video_processor = self.data_args.video_processor
        
        if self.soccer_dataset:
            cur_video_id = i
            video_path = self.soccer_video_list[cur_video_id]
            timestamp = self.timestamp_dict[cur_video_id]
            start_timestamp = self.start_timestamp_dict[cur_video_id]
            caption = self.caption_dict[cur_video_id]
            video_list = []
            for idx, start in enumerate(start_timestamp): 
                video = self.process_soccer_video(video_path=video_path, start_timestamp =start, end_timestamp = timestamp[idx], processor=video_processor,cur_fps=self.cur_fps)                    
                if video is None:
                    timestamp = timestamp[:idx]
                    start_timestamp = start_timestamp[:idx]
                    caption = caption[:idx]
                    break
                if idx>0:
                    video = video[1:]
                video_list.append(video)
            assert len(video_list) == len(timestamp), "Length of data error "
            if len(video_list) == 0:
                i = random.randint(0,len(self.soccer_video_list) - 1)
                return self.__getitem__(i) 

            if self.data_args.soccer_dataset_train_llm:
                data_dict = preprocess_llama_2_soccer(caption_data=caption,video_data= video_path,timestamp = timestamp, tokenizer=self.tokenizer, data_type = self.data_type)
            else:
                data_dict = preprocess_llama_2_soccer_cls(caption_data=caption,video_data= video_path,timestamp = timestamp, tokenizer=self.tokenizer, data_type = self.data_type)
            data_dict["video"] = video_list
            return data_dict

        elif self.ego4d_dataset:

            cur_video_id = i
            video_path = self.ego4d_video_list[cur_video_id]

            timestamp = self.timestamp_dict[cur_video_id]
            start_timestamp = self.start_timestamp_dict[cur_video_id]
            caption = self.caption_dict[cur_video_id]
            if self.data_args.soccer_dataset_train_llm:
                data_dict = preprocess_llama_2_ego4d(caption_data=caption,video_data=video_path ,timestamp = timestamp, tokenizer=self.tokenizer,data_type = self.data_type)
            else:
                data_dict = preprocess_llama_2_ego4d_cls(caption_data=caption,video_data=video_path ,timestamp = timestamp, tokenizer=self.tokenizer,data_type = self.data_type)
            video_path = ego_video_name_2_video_path(video_path)
            video_list = []
            for idx, start in enumerate(start_timestamp): 
                video = self.process_soccer_video(video_path=video_path, start_timestamp =start, end_timestamp = timestamp[idx], processor=video_processor,cur_fps=self.cur_fps)                    
                if video is None:
                    i = random.randint(0,len(self.ego4d_video_list) - 1)
                    return self.__getitem__(i)
                if idx>0:
                    video = video[1:]
                video_list.append(video)
            data_dict["video"] = video_list
            data_dict["video_path"] = video_path
            return data_dict
        elif self.live_dataset:
            anno = self.annos[i]
            video_path = anno["path"]
            if self.data_args.soccer_dataset_train_llm:
                data_dict = preprocess_llama_2_live(caption_data=anno["conversation"],video_data=video_path , tokenizer=self.tokenizer,data_type = self.data_type)
            else:
                data_dict = preprocess_llama_2_live_cls(caption_data=anno["conversation"],video_data=video_path , tokenizer=self.tokenizer,data_type = self.data_type)
            start_timestamp = anno["start_timestamp"]
            timestamp = anno["timestamp"]
            video_list = []
            for idx, start in enumerate(start_timestamp): 
                video = self.process_soccer_video(video_path=video_path, start_timestamp =start, end_timestamp = timestamp[idx], processor=video_processor,cur_fps=self.cur_fps)                    
                if video is None:
                    i = random.randint(0,len(self.annos) - 1)
                    return self.__getitem__(i)
                video_list.append(video)
            data_dict["video"] = video_list
            data_dict["video_path"] = video_path
            return data_dict 

        elif self.ego4d_lta_dataset:
            data = self.annos[i]
            if self.data_type == "train":
                data_dict = preprocess_llama_2_ego4d_lta( data=data, tokenizer=self.tokenizer,data_type = self.data_type)
            else:
                data_dict = preprocess_llama_2_ego4d_lta_val_generate( data=data, tokenizer=self.tokenizer,data_type = self.data_type)
            video_path = data["video_path"]
            video_path = ego_video_name_2_video_path(video_path)
            video_list = []
            video = self.process_soccer_video(video_path = video_path, start_timestamp = data["load_ranges"][0], end_timestamp = data["load_ranges"][1], processor = video_processor, cur_fps = self.cur_fps)                    

            if video is None:
                i = random.randint(0, len(self.annos) - 1)
                return self.__getitem__(i)
            video_list.append(video)
            data_dict["video"] = video_list
            data_dict["video_path"] = video_path
            return data_dict
        elif self.offline_dataset:
            sources = self.list_data_dict[i]
            if "video" in sources:
                num_frames = NUM_FRAMES if self.data_args.num_frames is None else self.data_args.num_frames
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                MODAL_list = []
                video_file = self.list_data_dict[i]['video']
                video_file = os.path.join(self.data_args.data_folder, video_file)
                try: 
                    video = process_video(video_file, video_processor, self.data_args.image_aspect_ratio, num_frames)
                except Exception as e:
                    traceback.print_exc()
                    backup_idx = random.randint(0, len(self.list_data_dict)-1)
                    print(f"Encounted error when reading video {video_file}, use {backup_idx}-th example instead!!!")
                    return self.__getitem__(backup_idx)

                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                MODAL_list.append('VIDEO')
                data_dict = preprocess_llama_2(sources, self.tokenizer, MODAL_list=MODAL_list,data_type = self.data_type)
                data_dict['video'] = [video]
                return data_dict
            elif "image" in sources:
                image_file = self.list_data_dict[i]['image']
                image_file = os.path.join(self.data_args.data_folder, image_file)
                image_processor = self.data_args.image_processor
                MODAL_list = []
                try:
                    image = process_image(image_file, image_processor, self.data_args.image_aspect_ratio)
                except Exception as e:
                    traceback.print_exc()
                    backup_idx = random.randint(0, len(self.list_data_dict)-1)
                    print(f"Encounted error when reading image {image_file}, use {backup_idx}-th example instead!!!")
                    return self.__getitem__(backup_idx)
                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
                MODAL_list.append("IMAGE")
                data_dict = preprocess_llama_2(sources, self.tokenizer, MODAL_list=MODAL_list,data_type = self.data_type)
                data_dict['image'] = [image]
                return data_dict
            else:
                backup_idx = random.randint(0, len(self.list_data_dict)-1)
                return self.__getitem__(backup_idx)
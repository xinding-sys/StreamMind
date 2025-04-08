# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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
from abc import ABC, abstractmethod

import einops
import torch
import torch.nn as nn

from .multimodal_projector import load_mm_projector
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector
from ..mm_utils import get_anyres_image_grid_shape
from ..constants import NUM_FRAMES, IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_MMODAL_PATCH_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN, MMODAL_TOKEN_INDEX


class Videollama2MetaModel:

    def __init__(self, config):
        super(Videollama2MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if os.path.exists(pretrain_mm_mlp_adapter):
                is_local = True
                if os.path.isdir(pretrain_mm_mlp_adapter):
                    mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            else:
                # Support loading projector weights from remote HuggingFace model hub
                is_local = False
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.replace('mm_projector.bin', '')
                pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_mlp_adapter)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            # self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            # set strict=False to avoid missing key error regarding bert.embeddings.position_ids
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class Videollama2MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def num_frames(self):
        if hasattr(self.config, 'num_frames'):
            return self.config.num_frames
        else:
            return NUM_FRAMES

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_or_videos(self, images_or_videos, modalities):
        import pdb
        pdb.set_trace()
        num_frames = self.config.num_frames if hasattr(self.config, 'num_frames') else NUM_FRAMES

        videos = [x.unsqueeze(0).expand(num_frames, -1, -1, -1) if modal == 'image' else x for x, modal in zip(images_or_videos, modalities)]
        videos = torch.stack(videos, dim=0)#[16,8,3,336,336]

        assert len(videos.size()) == 5
        batch_size = videos.size(0)

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')#[num_frames*batchsize,3,336,336][512,3,336,336]
        frames_features = self.get_model().get_vision_tower()(frames)#self.get_model()-> return self.model,使用clip做image encode,frames_features:[num_frames*batchsize,576,1024] [512,576,1024]
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)#[batch:16,num_frames:32,576,1024]

        return self.temporal_aggregator(frames_features)#这个就是过那个connector，见下




    def encode_images_or_videos_score(self, images_or_videos):
        # import pdb
        # pdb.set_trace()
        num_frames = images_or_videos.shape[0]
        videos = images_or_videos.unsqueeze(0)


        assert len(videos.size()) == 5
        batch_size = videos.size(0)
        # import pdb
        # pdb.set_trace()

        frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')#[num_frames*batchsize,3,336,336][512,3,336,336]
        #fps=50的时候gpu都会炸？？？？？？？
        if frames.shape[0] > 600:
            frames = frames[-600:]
        frames_features = self.get_model().get_vision_tower()(frames)#self.get_model()-> return self.model,使用clip做image encode,frames_features:[num_frames*batchsize,576,1024] [512,576,1024]
        frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)#[batch:16,num_frames:32,576,1024]

        return self.temporal_aggregator(frames_features)#这个就是过那个connector，见下

    def mamba_encode_images_or_videos_score(self, frames_features):
        return self.temporal_aggregator(frames_features)#这个就是过那个connector，见下



    @torch.no_grad()
    def encode_all_videos_score(self, images_or_videos, modalities):
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
        from decord import VideoReader, cpu
        from PIL import Image
        from videollama2.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image, process_score_video
        # import pdb
        # pdb.set_trace()
        import torch.distributed as dist
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        #下列列表len=1000，每张卡取250
        processor = self.get_model().get_vision_tower().image_processor

        target_filenames = ["1_224p.mkv", "2_224p.mkv"]

        # score_video_list = find_video_files("/home/v-dingxin/blob/MatchTime/features_video",target_filenames)
        score_video_list = find_video_files("/mnt/input/MatchTime/features_video",target_filenames)
        # /mnt/input/video_llm/datasets/videollava_sft/videochatgpt_tune
        local_batch = len(score_video_list)// world_size

        # score_video_list = ["/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv"]
        for video_path in score_video_list[rank * local_batch : (rank+1) * local_batch]:
            print("################encode video ########################")
            print(video_path)
            file_name = os.path.basename(video_path)
            half = file_name.split("_224p.mkv")[0]


            # video = process_score_video_full_frame(video_path, data_args.video_processor, data_args.image_aspect_ratio,model = model)
            if isinstance(video_path, str):
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0), num_threads=1) 
                duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())

                for start in range(0 , duration , 500):                    
                    frame_id_list = list(range(start , start + 500))
                    if start + 500 > duration:
                        frame_id_list = list(range(start ,duration))

                    video_data = decord_vr.get_batch(frame_id_list).asnumpy()
                    

                    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
                    images = [expand2square(image, tuple(int(x*255) for x in processor.image_mean)) for image in images]
                    video = processor.preprocess(images, return_tensors='pt')['pixel_values']

                    num_frames = video.shape[0]
                    videos = video.unsqueeze(0)

                    assert len(videos.size()) == 5
                    batch_size = videos.size(0)

                    frames = einops.rearrange(videos, 'b t c h w -> (b t) c h w')#[num_frames*batchsize,3,336,336][512,3,336,336]
                    #fps=50的时候gpu都会炸？？？？？？？
                    frames_features = self.get_model().get_vision_tower()(frames)#self.get_model()-> return self.model,使用clip做image encode,frames_features:[num_frames*batchsize,576,1024] [512,576,1024]
                    frames_features = einops.rearrange(frames_features, '(b t) n h -> b t n h', b = batch_size)#[batch:16,num_frames:32,576,1024]


                    #path save
                    encode_feature_new_path = video_path.replace("features_video", "features_video_encode_ddp")
                    encode_feature_new_path = os.path.dirname(encode_feature_new_path)
                    os.makedirs(encode_feature_new_path, exist_ok=True)
                    encode_feature_new_path = os.path.join(encode_feature_new_path, "{}_encode_feature_frame_{}_{}.pt".format(half,start,start+500))
                    torch.save(frames_features,encode_feature_new_path)
                    print("save feature to {}".format(encode_feature_new_path))

    

    def temporal_aggregator(self, frames_features):
        """Temporal aggregation of frame features.
        Args:
            frames_features (torch.Tensor): Frame features with shape (b, t, n, h).
        Returns:
            torch.Tensor: Video features with shape (b, n, h).
        """
        # TODO: improve the merging method.
        # *********** mean pooling *************
        if self.config.mm_projector_type == "mlp2x_gelu" or self.config.mm_projector_type == "linear":
            video_features = self.get_model().mm_projector(frames_features.mean(1))
        # *********** spatial convolution *************
        elif self.config.mm_projector_type == "spatial_conv":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** spatial pooling *************
        elif self.config.mm_projector_type == "spatial_pool":
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "tc_connector" in self.config.mm_projector_type or "tp_connector" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        # *********** time  ************
        elif "mamba" in self.config.mm_projector_type:
            video_features = self.get_model().mm_projector(frames_features)
        else:
            raise Exception(f"Unsupported projector type {self.config.mm_projector_type}!!!")
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        # import pdb
        # pdb.set_trace()
        vision_tower = self.get_vision_tower()# vision encode-> clip
        # NOTE: text-only situation
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
        # import pdb
        # pdb.set_trace()
        X_features = self.encode_images_or_videos(Xs, keys)#len(Xs)=16,sx["image"]=[3,336,336],xs["video"]=[32,3,336,336] ,x_feature:[16,576,4096],keys表示的是Xs的modality：video or image,x_features:[16,576,4096]
        #sample[16:batch,32:frames,4096]

        #上面这一步就是做encode和video_connector的计算，之后做了embeding后直接输入llm中
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0

      
        # replace image/video/audio tokens with pre-computed embeddings(数据集的格式是[{'from': 'human', 'value': '<video>\nCan you describe the video events?'}, {'from': 'gpt', 'value': 'The video primarily shows a group of bikers getting '}]])
        #这里的<video>要放前面用encoder计算的x_fearture 
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # cur_X_features = X_features[batch_idx]
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue
            # import pdb
            # pdb.set_trace()
            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []#上面是找到input_ids中表示“video”，“image”的位置
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            # X_index_inonesample = 0
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start])) #取input_ids到"video"or “image”的token，然后做embedding
                cur_new_input_embeds.append(cur_X_features)#***************这个list里面现在就是token + x_feature,接下来还要把剩下的input_ids embedding存进list里，然后cat起来形成新的input_ids************************* 
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])#label也是取到video这个token
                    cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))#中间插入一段和x_feature形状一致的mask，用来和加入x_feature的intput_ids对齐
                    cur_labels = cur_labels[X_token_start+1:]
                    #[tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                            # -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                            # -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                            # -100, -100, -100, -100, -100, -100, -100, -100], device='cuda:0'), tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                            # -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                            # -100, -100, -100, -100, -100, -100, -100, -100], device='cuda:0')]
                cur_X_idx += 1
                #接下来时“video”后面的token,继续这样处理，直到按照这个token全部分开，都存入cur_new_labels,cur_new_input_embeds里面
                cur_input_ids = cur_input_ids[X_token_start+1:] #
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            if cur_input_ids.numel() > 0:#将video token后面的也加入到embedlist和labellist里面
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
                # [tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                #         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                #         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                #         -100, -100, -100, -100, -100, -100, -100, -100], device='cuda:0'), tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                #         -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                #         -100, -100, -100, -100, -100, -100, -100, -100], device='cuda:0'), tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                #          -100,  -100,  -100,  -100,  -100,  -100,  -100, 12875, 28808,   415,
                #          3798,  4190,   264,  7548,  4973,  3233, 28725,   304,  2856,   905,
                #           460,  2598,  8711,   298,   272,  7555, 28723,  1387,   460, 14380,
                #          8102,  1287,   905,  7312,  1401,   396,  2698, 28725,   741,   905,
                #         14557,  1060,   264, 16513,   304, 22727,   297,  4154, 28723,   415,
                #           676,  7825,  1401, 14827,   395,  2663, 28725,  1312,  1309,  8711,
                #           298,   272,  7555, 28723, 28705,     2], device='cuda:0')]
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
 
        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            import pdb
            pdb.set_trace()
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            import pdb
            pdb.set_trace()
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:#把attention_mask 的形状也填充到和加入video_feature一致,同时注意attention——mask=False的id表示的是padding部分，其他部分(question+video+answer)都应该是true
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels#new_input_embeds:[16,766,4096],attention_mask:[16,766]

    def frame_sample(self, duration, mode='uniform', local_fps=None,num_frames = None):
        import numpy as np
        if mode == 'uniform':
            # Calculate the size of each segment from which a frame will be extracted
            seg_size = float(duration - 1) / num_frames

            frame_ids = []
            for i in range(num_frames):
                # Calculate the start and end indices of each segment
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                # Append the middle index of the segment to the list
                frame_ids.append((start + end) // 2)

            return frame_ids
            # NOTE: old version
            # return np.linspace(0, duration-1, num_frames, dtype=int)
        elif mode == 'fps':
            # assert local_fps is not None
            # segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
            segment_len = 2
            # return np.arange(segment_len // 2, duration, segment_len, dtype=int)
            return np.arange(1, duration, segment_len, dtype=int)
        else:
            raise ImportError(f'Unsupported frame sampling mode: {mode}')
    

    def prepare_inputs_labels_for_multimodal_score(
         self, input_ids, attention_mask, past_key_values, labels, X_modalities,timestamp,half,**kwargs    ):
        vision_tower = self.get_vision_tower()
        # NOTE: text-only situation
        import pdb
        pdb.set_trace()
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
        # X_features = self.encode_images_or_videos(Xs, keys)
        X_features = self.encode_images_or_videos_score(Xs)#len(Xs)=16,sx["image"]=[3,336,336],xs["video"]=[32,3,336,336] ,x_feature:[16,576,4096],keys表示的是Xs的modality：video or image,x_features:[16,576,4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # cur_X_features = X_features[batch_idx]
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue

            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            # X_index_inonesample = 0
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start])) 
                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])
                    cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[X_token_start+1:]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start+1:] 
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels



    def prepare_inputs_labels_for_multimodal_score_stream(
         self, input_ids, attention_mask, past_key_values, labels, X_modalities,timestamp,half,**kwargs    ):
        vision_tower = self.get_vision_tower()
        past_review_caption = kwargs["past_review_caption"]
        # NOTE: text-only situation
        # import pdb
        # pdb.set_trace()
        if vision_tower is None or X_modalities is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and vision_tower is not None and Xs is not None and input_ids.shape[1] == 1:
            #    attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        Xs, keys = X_modalities
        # print(Xs.shape,kwargs["past_review_caption"][0].shape)
        # X_features = self.encode_images_or_videos(Xs, keys)
        X_features = self.encode_images_or_videos_score(Xs)#len(Xs)=16,sx["image"]=[3,336,336],xs["video"]=[32,3,336,336] ,x_feature:[16,576,4096],keys表示的是Xs的modality：video or image,x_features:[16,576,4096]


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0
        # replace image/video/audio tokens with pre-computed embeddings
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # cur_X_features = X_features[batch_idx]
            if (torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1 
                continue

            X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0] 
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            
            # X_index_inonesample = 0
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:X_token_start])) 
                cur_new_input_embeds.append(self.get_model().embed_tokens(kwargs["past_review_caption"][0])) 
                cur_new_input_embeds.append(cur_X_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:X_token_start])
                    cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_new_labels.append(torch.full((self.get_model().embed_tokens(kwargs["past_review_caption"][0]).shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[X_token_start+1:]

                cur_X_idx += 1
                cur_input_ids = cur_input_ids[X_token_start+1:] 
                X_token_indices = torch.where(torch.any(torch.stack([cur_input_ids == MMODAL_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            # NOTE: one cur_new_input_embeds per each  
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        # import pdb
        # pdb.set_trace()
        # padding
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
              
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
            else:
                # import pdb
                # pdb.set_trace()
                attention_mask = torch.full((new_input_embeds.shape[0],new_input_embeds.shape[1]), True, device=new_input_embeds.device)
        return None, attention_mask, past_key_values, new_input_embeds, new_labels




    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings  = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg  = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:]  = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def initialize_MM_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            for modal in ['IMAGE', 'VIDEO', 'AUDIO']:
                tokenizer.add_tokens([DEFAULT_MMODAL_PATCH_TOKEN[modal.upper()]], special_tokens=True)
            # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = 0
            for modal in ['IMAGE', 'VIDEO', 'AUDIO']:
                num_new_tokens += tokenizer.add_tokens([DEFAULT_MMODAL_START_TOKEN[modal.upper()], DEFAULT_MMODAL_END_TOKEN[modal.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 6  # start/end tokens for image/video/audio
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

#    Copyright 2024 Alibaba DAMO Academy
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
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.regnet import RegStage
from timm.models.layers import LayerNorm, LayerNorm2d
from transformers import TRANSFORMERS_CACHE
from dataclasses import dataclass
from .ssm import VideoMamba

from pytorch_lightning import LightningModule
from videollama2.constants import IGNORE_INDEX

@dataclass
class SSMConfig:
    d_code = 1024
    d_model = 2048
    n_ssm = 1
    n_classes = 400
    lr = 1.4e-4
    lr_min = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.02
    scheduler = "plateau"


def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, "mm_projector.bin")):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(
            model_path, cache_dir=cache_dir, repo_type="model"
        )
        if not os.path.exists(os.path.join(folder, "mm_projector.bin")):
            # downloading from remote repo
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, "mm_projector.bin"), map_location="cpu")
    mm_projector_weights = {
        k: v.to(torch.float16) for k, v in mm_projector_weights.items()
    }
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def count_parameters(model):
    value =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return value * 4 / 1024 / 1024

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == "mamba":
        projector_model = Video_Mamba_seq(config)
        print(f"Trainable parameters: {count_parameters(projector_model)}MB")
        # import  pdb
        # pdb.set_trace()
        return projector_model 
    elif projector_type == "stc_connector":
        # import pdb
        # pdb.set_trace()
        # print("config.mm_hidden_size", config.mm_hidden_size)
        # print("config.hidden_size", config.hidden_size)
        projector_model = STCConnector(config)
        print(f"Trainable parameters: {count_parameters(projector_model)}MB")
        return projector_model
    elif projector_type == "stp_connector":
        return STPConnector(config)
    elif projector_type == "stc_connector_v35":
        return STCConnectorV35(config)
    elif projector_type == "spatial_conv":
        return SpatialConv(config)
    elif projector_type == "spatial_pool":
        return SpatialPool(config)
    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


class PreNet(nn.Module):
    def __init__(self, d_code, d_model):
        super(PreNet, self).__init__()
        self.fc3 = nn.Linear(d_code, d_model)

    def forward(self, x):
        x = self.fc3(x)
        x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.5)
        return x

class PostNet(nn.Module):
    def __init__(self, d_model, n_class):
        super(PostNet, self).__init__()
        self.fc3 = nn.Linear(d_model, n_class)

    def forward(self, x):
        x = F.leaky_relu(x)
        # x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x



class LinearBlock(nn.Module):
    def __init__(self, dim, expansion_factor=4, dropout=0.,norm_layer=LayerNorm):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim))
        # self.ln = nn.LayerNorm(dim)
        self.ln = norm_layer(dim)
    def forward(self, x):
        return x + self.fn(self.ln(x))





class TextProj(nn.Module):
    def __init__(self, embedding_dim=4096, output_dim=512, norm_layer=LayerNorm):
        super().__init__()
        # self.ln_final = norm_layer(embedding_dim)
        self.embedding_dim = embedding_dim
        # self.text_projection = nn.Parameter(torch.empty(embedding_dim, output_dim))
        expansion_factor = 2
        dropout = 0
        proj_bias = True
        num_layers_text = 4
        self.text_adaptor = nn.Sequential(
            *[LinearBlock(embedding_dim, expansion_factor, dropout) for _ in range(num_layers_text)],
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_dim, bias=proj_bias),
            )
        # self.text_adaptor =  nn.Linear(embedding_dim, output_dim, bias=proj_bias)
        self.grad_checkpointing = False
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def lock(self, unlocked_layers, freeze_layer_norm):
        for param in self.text_adaptor.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, text, return_all_features: bool=False):
        x = self.text_adaptor(text)
        # x = self.ln_final(text)
        # if not return_all_features:
        #     x = x @ self.text_projection
        return x


# class ClsNet(nn.Module):
#     def __init__(self, d_model, depth, n_class ):
#         super(ClsNet, self).__init__()
#         self.cls = nn.Linear(d_model, n_class)
#     def forward(self, x,past_caption= None):
#         x = self.cls(x)
#         return x



        
# class ClsNet(nn.Module):
#     def __init__(self, d_model, depth, n_class ):
#         super(ClsNet, self).__init__()
#         self.blocks = nn.ModuleList([
#             LinearBlock(d_model) for _ in range(depth)
#         ])
#         self.cls = nn.Linear(d_model, n_class)
#         self.pooling = nn.AdaptiveAvgPool1d(1)
#         # self.Softmax = nn.Softmax(dim=1)
#     def forward(self, x,past_caption= None):
#         # x = F.dropout(x, p=0.5)
#         # import pdb
#         # pdb.set_trace()
#         if past_caption is not None:
#             x = torch.cat((past_caption,x),dim = 1)
#         for block in self.blocks:
#             x = block(x)
#         # x = x.transpose(1, 2)
#         # x = self.pooling(x)
#         # x = x.squeeze(-1)
#         # x = x[:,-1,:]
#         x = self.cls(x)
#         # x = self.Softmax(x)
#         return x


from transformers import MistralConfig, MistralForCausalLM,Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MistralForCausalLM_cls(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # self.loss_fct = FocalLoss(alpha=0.45)
        # import pdb
        # pdb.set_trace()
        # self.cls_net = nn.Linear(config.vocab_size+2,2)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # import pdb
        # pdb.set_trace()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        # logits = self.cls_net(logits)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # import pdb
            # pdb.set_trace()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_logits = shift_logits.view(-1, 2)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            # shift_logits = shift_logits[shift_labels!=IGNORE_INDEX]
            # shift_labels = shift_labels[shift_labels!=IGNORE_INDEX]
            # loss_fct = CrossEntropyLoss(weight=torch.tensor([0.15,0.85]).to(shift_logits.device),label_smoothing = 0.2)
            weight_list = [1]* (self.config.vocab_size-2)
            weight_list.append(0.15)
            weight_list.append(0.85)
            loss_fct = CrossEntropyLoss(weight=torch.tensor(weight_list).to(shift_logits.device))
            # loss_fct = CrossEntropyLoss(weight=torch.tensor([0.01,0.99]).to(shift_logits.device))
            # loss_fct = CrossEntropyLoss()
            # loss = self.loss_fct(shift_logits, shift_labels)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ClsNet(nn.Module):
    def __init__(self, d_model, depth, n_class ):
        super(ClsNet, self).__init__()
        mis_config = MistralConfig()
        # import pdb
        # pdb.set_trace()
        mis_config.vocab_size = 2
        mis_config.num_hidden_layers = 4
        self.cls_model = MistralForCausalLM_cls(config=mis_config)
        # self.cls_model.resize_token_embeddings(mis_config.vocab_size + 2) #加了两个specific token:</silence> </response> 
        # self.Softmax = nn.Softmax(dim=1)
    def forward(self, x, cls_labels, cls_attention_mask):
        x = self.cls_model(inputs_embeds = x,
                labels = cls_labels,
                attention_mask = cls_attention_mask)
        return x

import time
import os

class Video_Mamba_seq(LightningModule):
    def __init__(self, model_config):
        super(Video_Mamba_seq, self).__init__()
        self.pre_net = PreNet(model_config.mm_hidden_size, model_config.hidden_size)
        mamba_config = SSMConfig()
        mamba_config.d_code=model_config.hidden_size
        mamba_config.d_model=model_config.hidden_size
        self.mamba_model = VideoMamba(mamba_config)
        self.post_net = PostNet(model_config.hidden_size, model_config.hidden_size)
        self.cls_net = ClsNet( d_model=model_config.hidden_size, depth=4, n_class=2)
        # self.cls_net = None
        self.time_list = []
        self.videoid = 0 
    def forward(self, x, cls_inference = False,cls_training = False,cls_demo = False,frames_features_shape = [],prompt_time_input_ids = None,prompt_time_lable = None):
        b, t, l, d = x.shape
        x = torch.mean(x, dim=2) 
        x = einops.rearrange(x, "b t d -> (b t) d", b=b, t=t)
        # import pdb
        # pdb.set_trace()
        x = self.pre_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)
        x = self.mamba_model(x)
        x = einops.rearrange(x, "b t d -> (b t) d")
        x = self.post_net(x)
        x = einops.rearrange(x, "(b t) d -> b t d", b=b, t=t)
        # import pdb
        # pdb.set_trace()
        # self.videoid += 1
        # os.mkdir("/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}".format(self.videoid))
        # torch.save(x,"/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}/cur_frame_feature.pt".format(self.videoid))
        
        if cls_training or cls_inference:
            import pdb
            pdb.set_trace()
            if prompt_time_input_ids is not None and prompt_time_input_ids.numel()>1:
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                X_prompt_indices = torch.where(prompt_time_input_ids == -201)[1]
                X_prediction_indices = torch.where(prompt_time_lable == 32000)[1] #找到</silence>这个token的位置，</response>是32001
                #[INST] <<SYS>>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<</SYS>>\n\nWhat are the prerequisites for the next task?<video>\n [/INST] </silence> </s>"
                prompt_template_inputs = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[: , : X_prompt_indices])#这个是sys那部分
                prompt_template_inputs_requirements = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[ : , X_prompt_indices + 1 : X_prediction_indices])#这个是用户需求
                prompt_template_inputs_rest = self.cls_net.cls_model.model.embed_tokens(prompt_time_input_ids[ : , X_prediction_indices + 1 :])

                prompt_template_labels = prompt_time_lable[:,:X_prompt_indices].to(x.device)
                prompt_template_labels_requirements = prompt_time_lable[:,X_prompt_indices + 1 : X_prediction_indices].to(x.device)
                prompt_template_labels_rest = prompt_time_lable[:,X_prediction_indices + 1:].to(x.device)
                
                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device)).unsqueeze(0)
                caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device)).unsqueeze(0)
                input_embeds = []
                cls_labels = []
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([prompt_template_inputs,
                                                    frame.unsqueeze(0).unsqueeze(0),
                                                    prompt_template_inputs_requirements,
                                                    eos_target,
                                                    prompt_template_inputs_rest], dim=1) for frame in cur_frame_feature[:-1]])

                        cls_label= torch.cat([torch.cat([prompt_template_labels,
                                                    torch.full((1,1),IGNORE_INDEX).to(x.device),
                                                    prompt_template_labels_requirements,
                                                    torch.tensor([[32000]]).to(x.device),
                                                    prompt_template_labels_rest], dim=1) for _ in cur_frame_feature[:-1]])
                        input_embeds.append(input_embed)
                        cls_labels.append(cls_label)
                    input_embeds.append(torch.cat([prompt_template_inputs,
                                                    cur_frame_feature[-1].unsqueeze(0).unsqueeze(0),
                                                    prompt_template_inputs_requirements,
                                                    caption_target,
                                                    prompt_template_inputs_rest],dim=1))
                    
                    cls_labels.append(torch.cat([prompt_template_labels,
                                                torch.full((1,1),IGNORE_INDEX).to(x.device),
                                                prompt_template_labels_requirements,
                                                torch.tensor([[32001]]).to(x.device),
                                                prompt_template_labels_rest],dim = 1))

                input_embed = torch.cat(input_embeds)
                cls_label= torch.cat(cls_labels)
                # input_embed =  einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
                # cls_label =  einops.rearrange(cls_labels, "(b t)  -> b t ", t=2)
                if cls_label.shape[0]>4000:
                    cls_label = cls_label[:4000]
                    input_embed = input_embed[:4000]
                #all frame
                # input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
                # cls_label = torch.nn.utils.rnn.pad_sequence(cls_labels,batch_first=True,padding_value=IGNORE_INDEX)

                #mask
                cls_attention_mask = input_embed.ne(pad_token_id)

                if cls_training:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)
                # cls_loss = self.cls_net(x,None)
                    return cls_output
                else:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)

                    return cls_output,cls_label


            else:
                pad_token_id = 0
                input_embeds = []
                cls_labels = []
                start_feature_idx = [0] + frames_features_shape[:-1]
                for idx, end_frame_idx in enumerate(frames_features_shape):
                    cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                    # torch.save(cur_frame_feature,"/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}/cur_frame_feature_{}.pt".format(self.videoid,idx))

                    eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                    caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device))
                    # ignore_tensor = torch.full(cur_frame_feature[0].unsqueeze(0).shape,IGNORE_INDEX).to(x.device).to(caption_target.dtype)
                    ignore_tensor = torch.tensor([IGNORE_INDEX]).to(x.device)

                    if cur_frame_feature.shape[0] > 1:
                        input_embed = torch.cat([torch.cat([frame.unsqueeze(0),eos_target]) for frame in cur_frame_feature[:-1]])
                        eos_label = torch.cat([torch.cat([ignore_tensor, torch.tensor([0]).to(x.device)]) for _ in cur_frame_feature[:-1]])

                        input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0), caption_target]))
                        cls_labels.append(torch.cat([eos_label, ignore_tensor, torch.tensor([1]).to(x.device)]))
                    else:
                        input_embed = torch.cat([cur_frame_feature,caption_target])
                        caption_label = torch.cat([ignore_tensor, torch.tensor([1]).to(x.device)])
                        input_embeds.append(input_embed)
                        cls_labels.append(caption_label)
                #last frame
                # import pdb
                # pdb.set_trace()
                input_embeds = torch.cat(input_embeds)
                cls_labels = torch.cat(cls_labels)
                input_embed =  einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
                cls_label =  einops.rearrange(cls_labels, "(b t)  -> b t ", t=2)
                if cls_label.shape[0]>4000:
                    cls_label = cls_label[:4000]
                    input_embed = input_embed[:4000]
                #all frame
                # input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
                # cls_label = torch.nn.utils.rnn.pad_sequence(cls_labels,batch_first=True,padding_value=IGNORE_INDEX)

                #mask
                cls_attention_mask = input_embed.ne(pad_token_id)

                if cls_training:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)
                # cls_loss = self.cls_net(x,None)
                    return cls_output
                else:
                    cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)

                    return cls_output,cls_label
            
        if cls_demo:
            # import pdb
            # pdb.set_trace()
            pad_token_id = 0
            input_embeds = []
            # start_feature_idx = [0] + frames_features_shape[:-1]
            input_embeds.append(x[0][-1].unsqueeze(0))
            input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
            cls_attention_mask = input_embed.ne(pad_token_id)

            # start = time.time()
            cls_output = self.cls_net(input_embed, cls_labels = None, cls_attention_mask = cls_attention_mask)


            
            return x , cls_output.logits[0][-1]

        return x

def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class STCConnector(nn.Module):
    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        """Temporal Convolutional Vision-Language Connector.

        Args:
            config: config object.
            downsample: (temporal, height, width) downsample rate.
            depth: depth of the spatial interaction blocks.
            mlp_depth: depth of the vision-language projector layers.
        """
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = config.mm_hidden_size
        self.hidden_size = hidden_size = config.hidden_size
        self.output_hidden_size = output_hidden_size = config.hidden_size
        # TODO: make these as config arguments
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.downsample = downsample
        if depth != 0:
            self.s1 = RegStage(
                depth=depth,
                in_chs=encoder_hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s1 = nn.Identity()
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=1,
                bias=True,
            ),
            nn.SiLU(),
        )
        if depth != 0:
            self.s2 = RegStage(
                depth=depth,
                in_chs=hidden_size,
                out_chs=hidden_size,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
        else:
            self.s2 = nn.Identity()
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        self.cls_net = ClsNet( d_model=hidden_size, depth=4, n_class=2)
        # self.cls_net = None
        self.time_list = []
        self.videoid = 0 
    # def forward(self, x):
    def forward(self, x, cls_inference = False,cls_training = False,cls_demo = False,frames_features_shape = []):

        """Aggregate tokens on the temporal and spatial dimensions.
        Args:
            x: input tokens [b, t, h, w, d] / [b, t, l, d]
        Returns:
            aggregated tokens [b, l, d]
        """
        import pdb
        pdb.set_trace()
        t = x.size(1)
        if x.ndim == 4:
            hw = int(x.size(2) ** 0.5)
            x = einops.rearrange(x, "b t (h w) d -> b d t h w", h=hw, w=hw)
        elif x.ndim == 5:
            x = einops.rearrange(x, "b t h w d -> b d t h w")

        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        # 1. the first stage of the adapter
        x = self.s1(x)
        x = einops.rearrange(x, "(b t) d h w -> b d t h w", t=t)
        # 2. downsampler
        x = self.sampler(x)
        new_t = x.size(2)
        # 3. the second stage of the adapter
        x = einops.rearrange(x, "b d t h w -> (b t) d h w")
        x = self.s2(x)
        x = einops.rearrange(x, "(b t) d h w -> b (t h w) d", t=new_t)
        x = self.readout(x)
        import pdb
        pdb.set_trace()

        self.videoid += 1
        os.makedirs("/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}".format(self.videoid),exist_ok = True)
        torch.save(x,"/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}/cur_frame_feature.pt".format(self.videoid))
        
        if cls_training or cls_inference:
            pad_token_id = 0
            input_embeds = []
            cls_labels = []
            start_feature_idx = [0] + frames_features_shape[:-1]
            for idx, end_frame_idx in enumerate(frames_features_shape):
                cur_frame_feature = x[0][start_feature_idx[idx] : end_frame_idx]
                torch.save(cur_frame_feature,"/home/v-dingxin/videollama2_plus-main/paper/immediate_memory_stc_{}/cur_frame_feature_{}.pt".format(self.videoid,idx))

                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                caption_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([1]).to(x.device))
                # ignore_tensor = torch.full(cur_frame_feature[0].unsqueeze(0).shape,IGNORE_INDEX).to(x.device).to(caption_target.dtype)
                ignore_tensor = torch.tensor([IGNORE_INDEX]).to(x.device)

                if cur_frame_feature.shape[0] > 1:
                    input_embed = torch.cat([torch.cat([frame.unsqueeze(0),eos_target]) for frame in cur_frame_feature[:-1]])
                    eos_label = torch.cat([torch.cat([ignore_tensor, torch.tensor([0]).to(x.device)]) for _ in cur_frame_feature[:-1]])

                    input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0), caption_target]))
                    cls_labels.append(torch.cat([eos_label, ignore_tensor, torch.tensor([1]).to(x.device)]))
                else:
                    input_embed = torch.cat([cur_frame_feature,caption_target])
                    caption_label = torch.cat([ignore_tensor, torch.tensor([1]).to(x.device)])
                    input_embeds.append(input_embed)
                    cls_labels.append(caption_label)
            #last frame
            # import pdb
            # pdb.set_trace()
            input_embeds = torch.cat(input_embeds)
            cls_labels = torch.cat(cls_labels)
            input_embed =  einops.rearrange(input_embeds, "(b t) c -> b t c", t=2)
            cls_label =  einops.rearrange(cls_labels, "(b t)  -> b t ", t=2)
            if cls_label.shape[0]>4000:
                cls_label = cls_label[:4000]
                input_embed = input_embed[:4000]
            #all frame
            # input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
            # cls_label = torch.nn.utils.rnn.pad_sequence(cls_labels,batch_first=True,padding_value=IGNORE_INDEX)

            #mask
            cls_attention_mask = input_embed.ne(pad_token_id)

            if cls_training:
                cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)
            # cls_loss = self.cls_net(x,None)
                return cls_output
            else:
                cls_output= self.cls_net(input_embed,cls_labels=cls_label,cls_attention_mask=cls_attention_mask)

                return cls_output,cls_label
            
        if cls_demo:
            # import pdb
            # pdb.set_trace()
            pad_token_id = 0
            input_embeds = []
            start_feature_idx = [0] + frames_features_shape[:-1]
            if len(frames_features_shape) == 0:
                input_embeds.append(x[0])
            else:
                cur_frame_feature = x[0][frames_features_shape[-1]:]
                eos_target = self.cls_net.cls_model.model.embed_tokens(torch.tensor([0]).to(x.device))
                if cur_frame_feature.shape[0] > 1:
                    input_embed = torch.cat([torch.cat([frame.unsqueeze(0),eos_target]) for frame in cur_frame_feature[:-1]])
                    input_embeds.append(torch.cat([input_embed, cur_frame_feature[-1].unsqueeze(0)]))
                else:
                    input_embeds.append(cur_frame_feature)

                # input_embeds.append(x[0][frames_features_shape[-1]:])
            input_embed = torch.nn.utils.rnn.pad_sequence(input_embeds,batch_first=True,padding_value=pad_token_id)
            cls_attention_mask = input_embed.ne(pad_token_id)

            # start = time.time()
            cls_output = self.cls_net(input_embed, cls_labels = None, cls_attention_mask = cls_attention_mask)
            # end = time.time()
            # self.time_list.append(end - start)
            # print("cls_net reponse time:{} s".format(sum(self.time_list)/len(self.time_list)))
            return x , cls_output.logits[0][-1]


        return x


class STPConnector(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )
        self.sampler = nn.Sequential(nn.AvgPool3d(downsample), nn.SiLU())


class STCConnectorV35(STCConnector):

    def __init__(self, config, downsample=(2, 2, 2), depth=4, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )
        self.sampler = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                kernel_size=downsample,
                stride=downsample,
                padding=0,
                bias=True,
            ),
            nn.SiLU(),
        )


class SpatialConv(STCConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )


class SpatialPool(STPConnector):

    def __init__(self, config, downsample=(1, 2, 2), depth=0, mlp_depth=2):
        super().__init__(
            config=config, downsample=downsample, depth=depth, mlp_depth=mlp_depth
        )

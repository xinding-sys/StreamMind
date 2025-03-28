# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..videollama2_arch import Videollama2MetaModel, Videollama2MetaForCausalLM


class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"


class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)


class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.model = Videollama2MistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        import pdb
        pdb.set_trace()
        if inputs_embeds is None:
            if "timestamp" in kwargs.keys():
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                    input_ids,#[16,191]
                    attention_mask,#[16,191]
                    past_key_values,
                    labels,#[16,191,]
                    images,#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                    **kwargs
                )
            else:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,#[16,191]
                    attention_mask,#[16,191]
                    past_key_values,
                    labels,#[16,191,]
                    images#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                )

        return super().forward(#这个做的就是mistral的forward
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        score_video = kwargs.pop("score_video",None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        import pdb
        pdb.set_trace()
        if score_video :
            (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            _) = self.prepare_inputs_labels_for_multimodal_score(
                    input_ids=inputs,#[16,191]
                    attention_mask=attention_mask,#[16,191]
                    past_key_values=None,
                    labels=None,#[16,191,]
                    X_modalities=[images_or_videos,["video"]],#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                    timestamp=None,anonymized=None,half=None
                )

        else:

            if images_or_videos is not None:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    X_modalities=[images_or_videos, modal_list],
                    
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("videollama2_mistral", Videollama2MistralConfig)
AutoModelForCausalLM.register(Videollama2MistralConfig, Videollama2MistralForCausalLM)

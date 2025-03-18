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

import time

class Videollama2MistralConfig(MistralConfig):
    model_type = "videollama2_mistral"


class Videollama2MistralModel(Videollama2MetaModel, MistralModel):
    config_class = Videollama2MistralConfig

    def __init__(self, config: MistralConfig):
        super(Videollama2MistralModel, self).__init__(config)

from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 2, size_average=True):
        """
        FocalLoss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 FocalLoss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retina net中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retina net中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)

            # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
            self.alpha[0].fill_(alpha)
            self.alpha[1:].fill_(1-alpha)

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, preds, labels):
        """
        FocalLoss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        # import pdb
        # pdb.set_trace()
        ###############################
        # 一、初始操作
        ###############################

        # 按照最后一个维度重新调整矩阵形状，因为最后一个维度是分类数
        preds = preds.view(-1,preds.size(-1))

        alpha = self.alpha.to(preds.device)

        ###############################
        # 二、计算预测概率Pt
        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        ###############################

        # 将 preds 张量在第 1 个维度上执行 softmax 操作，过softmax之后的，就是pt
        pt = preds_softmax = F.softmax(preds, dim=1)
        # 交叉熵损失函数 CELoss(pt) = -log(pt)，这个pt，就是预估值，多分类是softmax后的概率值，二分类是sigmoid后的值
        # 在softmax后面接着log，这样算交叉熵只用前面加个负号
        log_pt = preds_logSoftmax = torch.log(pt)

        ###############################
        # 三、选真实标签对应的数据
        ###############################

        # labels.view(-1,1) 是将 labels 张量的形状调整为 (N, 1)
        # Ensure the labels are long, not float
        labelsView = labels.view(-1, 1).long()
        # 下面操作的目的就是选出真实标签对应的pt
        pt = pt.gather(1,labelsView)
        # 下面操作的目的就是选出真实标签对应的log_pt
        log_pt = log_pt.gather(1,labelsView)

        ###############################
        # 四、不带α的focal-loss
        ###############################

        # focalLoss(pt) = -(1-pt)^γ * log(pt)
        loss = -torch.mul(torch.pow((1-pt), self.gamma), log_pt)


        ###############################
        # 五、带上α的focal-loss
        ###############################
        # labels.view(-1) 的作用是将 labels 张量的形状调整为一个一维张量
        label_flatten=labelsView.view(-1)
        # 因为算softmax的时候，会把所有的值的softmax都算出来，然而我们需要的只要真实标签的那些而已
        # 所以要进行取值操作
        # 整句话的作用就是alpha根据label值，取到每个分类对应的数值α
        alpha = alpha.gather(0,label_flatten)
        # 损失乘上α向量操作
        loss = torch.mul(alpha, loss.t())


        # 根据需求，看损失是求平均还是求和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class Videollama2MistralForCausalLM(MistralForCausalLM, Videollama2MetaForCausalLM):
    config_class = Videollama2MistralConfig

    def __init__(self, config, **kwargs):
        super(MistralForCausalLM, self).__init__(config)
        self.train_iteration = 0
        self.model = Videollama2MistralModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.frame_feature = None
        self.past_review_caption = None
        self.past_review_caption_list = []
        self.interval_id_list = []
        self.loss_fct = CrossEntropyLoss()
        # self.focal_loss = FocalLoss(alpha= 0.25)
        self.time_list = []
        self.sample_per = kwargs.pop("sample_per", 0.5)
        self.sample_type = kwargs.pop("sample_type","ssss")


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
        cls_output = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            if "timestamp" in kwargs.keys():
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    cls_output
                ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                    input_ids,#[16,191]
                    attention_mask,#[16,191]
                    past_key_values,
                    labels,#[16,191,]
                    images,#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                    sample_per = self.sample_per,
                    sample_type = self.sample_type,
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

        # import pdb
        # pdb.set_trace()
        # if labels is None and inputs_embeds is None and attention_mask is None:
        if cls_output is not None:
            return cls_output
            # return {"logits":None,
            #         "loss":cls_loss}
        else:
            # import pdb
            # pdb.set_trace()
            # start = time.time()
            llm_output = super().forward(#这个做的就是mistral的forward
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        labels=labels,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict)
            # end = time.time()
            # print("LLM single reponse time:{} s".format(end - start))


            # print("llm")
            # if "loss" in llm_output:#训练的时候用
            #     # print("llm")
            #     llm_output["loss"] += cls_loss
        # import pdb
        # pdb.set_trace()
        # return {"llm_output":llm_output,
        #             "llm_labels":labels}
        llm_eval = kwargs.pop("llm_eval",None)
        if llm_eval:
            return llm_output,labels
        return llm_output

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
        # import pdb
        # pdb.set_trace()
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
                    _,_
                ) = self.prepare_inputs_labels_for_multimodal_score_stream(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    timestamp = None,
                    model_type= "llm",
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

    @torch.no_grad()
    def stream_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        score_video = kwargs.pop("score_video",None)
        tokenizer = kwargs.pop("tokenizer",None)

        # import pdb
        # pdb.set_trace()
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        
        if self.past_review_caption is not None:
            past_caption_ids = tokenizer(
                                self.past_review_caption,
                                return_tensors="pt",
                                padding="longest",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                ).input_ids
        else:
            past_caption_ids = None

        (
        input_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        _,
        cls_pred,
        self.frame_feature ) = self.prepare_inputs_labels_for_multimodal_score_stream_inference(
                input_ids=inputs,#[16,191]
                attention_mask=attention_mask,#[16,191]
                past_key_values=None,
                labels=None,#[16,191,]
                X_modalities=[images_or_videos,["video"]],#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                past_review_caption = past_caption_ids,
                frames_features = self.frame_feature
            )
        if cls_pred == 0:
            return None

        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
  
        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

        if self.past_review_caption is None :
            self.past_review_caption = output 
        else:
            self.past_review_caption += output

        return output


       
    @torch.no_grad()
    def stream_generate_demo(
        self,
        inputs: Optional[torch.Tensor] = None,
        images_or_videos: Optional[torch.Tensor] = None,
        modal_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        score_video = kwargs.pop("score_video",None)
        tokenizer = kwargs.pop("tokenizer",None)

        # import pdb
        # pdb.set_trace()
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        (
        input_ids,
        attention_mask,
        past_key_values,
        inputs_embeds,
        _,
        cls_pred,
        self.frame_feature,
        interval_id ) = self.prepare_inputs_labels_for_multimodal_score_stream_inference_demo(
                input_ids=inputs,#[16,191]
                attention_mask=attention_mask,#[16,191]
                past_key_values=None,
                labels=None,#[16,191,]
                X_modalities=[images_or_videos,["video"]],#len = 2 len(image[0])=16,image[0][0].shape = [8,3,336,336]
                frames_features = self.frame_feature,
                interval_id_list = self.interval_id_list
            )
        #"<video>\n [/INST]"

        if cls_pred == 0:
            return None,cls_pred
        # else:
            # self.interval_id_list.append(interval_id)
        # start = time.time()
        output_idx = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        # end = time.time()
        # self.time_list.append(end - start)

        # print("LLM reponse time:{} s".format(sum(self.time_list)/len(self.time_list)))

        output = tokenizer.batch_decode(output_idx, skip_special_tokens=True)[0].strip()

        return output,cls_pred


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

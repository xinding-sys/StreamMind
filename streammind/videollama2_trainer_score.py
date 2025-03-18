# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    TRAINER_STATE_NAME,
)

# from transformers.trainer import (
#     is_sagemaker_mp_enabled,
#     get_parameter_names,
#     has_length,
#     ALL_LAYERNORM_LAYERS,
#     logger,
#     TRAINER_STATE_NAME,
#     DebugOption,
#     deepspeed_init,
#     deepspeed_load_checkpoint, 
#     is_deepspeed_available,
#     TrainerState,
#     get_model_param_count,
#     hp_params,
#     get_dataloader_sampler,
#         ADAPTER_CONFIG_NAME,
#     ADAPTER_SAFE_WEIGHTS_NAME,
#     ADAPTER_WEIGHTS_NAME,
#     CONFIG_NAME,
#     SAFE_WEIGHTS_INDEX_NAME,
#     SAFE_WEIGHTS_NAME,
#     WEIGHTS_INDEX_NAME,
#     WEIGHTS_NAME,
#     XLA_FSDPV2_MIN_VERSION,
#     PushInProgress,
#     PushToHubMixin,
#     can_return_loss,
#     find_labels,
#     is_accelerate_available,
#     is_apex_available,
#     is_bitsandbytes_available,
#     is_datasets_available,
#     is_galore_torch_available,
#     is_in_notebook,
#     is_ipex_available,
#     is_peft_available,
#     is_safetensors_available,
#     is_sagemaker_dp_enabled,
#     is_sagemaker_mp_enabled,
#     is_torch_compile_available,
#     is_torch_mlu_available,
#     is_torch_neuroncore_available,
#     is_torch_npu_available,
#     is_torch_xla_available,
#     logging,
#     strtobool,
#     speed_metrics,

# )
from tqdm.auto import tqdm


from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    GradientAccumulationPlugin,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from concurrent.futures import ThreadPoolExecutor
import os
import time
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)





def load_video_segment(video_encode_path, input_device, start_idx, end_idx, segment):
    if os.path.exists(video_encode_path):
        return torch.load(video_encode_path, map_location=input_device)[:, start_idx:end_idx:segment]
    return None



class VideoLLaMA2Trainer(Trainer):
    def __init__(self, data_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.tlsd = tsld_loss
        self.data_args = data_args        
        video_processor = self.data_args.video_processor

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            self.args.distributed_state.wait_for_everyone()
        else:
            # NOTE: Supporting save complete lora checkpoint during training.
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                if self.args.local_rank == 0 or self.args.local_rank == -1:
                    state_dict = get_peft_state_maybe_zero_3(self.model.named_parameters(), self.args.lora_bias)
                    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
                    # save for acquring `config.json`
                    self.model.config.save_pretrained(output_dir)
                    # save for acquring `adapter_config.json`, `adapter_model.bin`
                    # self.model.save_pretrained(output_dir, state_dict=state_dict)
                    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

                # save for acquring lora adapter parameters & trainer states: `adapter_config.json`, `adapter_model.safetensors`
                super(VideoLLaMA2Trainer, self)._save_checkpoint(model, trial, metrics)
            else:
                super(VideoLLaMA2Trainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(VideoLLaMA2Trainer, self)._save(output_dir, state_dict)
    


    def video_timestamp_to_video(self,video_path,timestamp,half,fps=2,input_device=None):
        # if input_device is None:
        input_device = "cpu"
        video_fps = 25
        segment = video_fps//fps
        frame_stamp = 25 * timestamp
        # import pdb
        # pdb.set_trace()
        video_encode_list = []
        video_dir = os.path.dirname(video_path)
        video_encode_feature_dir = video_dir.replace("features_video", "features_video_encode_ddp")

        if (frame_stamp//500) == 0:
            video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,0,500))
            if frame_stamp + 100 < 500:
                video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp+100:segment])
            else:
                video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp:segment])
        else:
            for video_encode_id in range(frame_stamp//500):
                video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,video_encode_id*500,(video_encode_id+1)*500))
                if os.path.exists(video_encode_path):
                    frame_id = frame_stamp % 500
                    if video_encode_id == (frame_stamp//500) - 1:
                        if frame_stamp % 500 + 100 < 500:
                            video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp % 500+100:segment])
                        else:
                            video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,0:frame_stamp % 500:segment])
                    else:
                        video_encode_list.append(torch.load(video_encode_path,map_location=input_device)[:,::segment])
                else:
                    continue
        torch.cuda.empty_cache()
        return video_encode_list





    def video_timestamp_to_video_ddp(self, video_path, timestamp, half, fps=2, input_device="cpu"):
        video_fps = 25
        segment = video_fps // fps
        frame_stamp = 25 * timestamp
        video_encode_list = []
        video_dir = os.path.dirname(video_path)
        video_encode_feature_dir = video_dir.replace("features_video", "features_video_encode_ddp")

        # Parallel loading setup
        tasks = []
        with ThreadPoolExecutor() as executor:
            if (frame_stamp // 500) == 0:
                video_encode_path = os.path.join(video_encode_feature_dir, "{}_encode_feature_frame_{}_{}.pt".format(half, 0, 500))
                end_idx = frame_stamp + 100 if frame_stamp + 100 < 500 else frame_stamp
                tasks.append(executor.submit(load_video_segment, video_encode_path, input_device, 0, end_idx, segment))
            else:
                for video_encode_id in range(frame_stamp // 500):
                    video_encode_path = os.path.join(video_encode_feature_dir, "{}_encode_feature_frame_{}_{}.pt".format(half, video_encode_id * 500, (video_encode_id + 1) * 500))
                    frame_id = frame_stamp % 500
                    if video_encode_id == (frame_stamp // 500) - 1:
                        end_idx = frame_id + 100 if frame_id + 100 < 500 else frame_id
                        tasks.append(executor.submit(load_video_segment, video_encode_path, input_device, 0, end_idx, segment))
                    else:
                        tasks.append(executor.submit(load_video_segment, video_encode_path, input_device, 0, None, segment))

            for task in tasks:
                result = task.result()
                if result is not None:
                    video_encode_list.append(result)
        torch.cuda.empty_cache()
        return video_encode_list



    def video_timestamp_to_video_test(self,video_path,timestamp,half,fps=2,input_device=None):
        # if input_device is None:
        input_device = "cpu"
        video_fps = 25
        segment = video_fps//fps
        frame_stamp = 25 * timestamp
        # import pdb
        # pdb.set_trace()
        video_encode_list = []
        video_dir = os.path.dirname(video_path)
        video_encode_feature_dir = video_dir.replace("features_video", "features_video_encode_ddp")

        if (frame_stamp//500) == 0:
            video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,0,500))
            if frame_stamp + 100 < 500:
                video_encode_list.append(torch.tensor([1]))
            else:
                video_encode_list.append(torch.tensor([1]))
        else:
            for video_encode_id in range(frame_stamp//500):
                video_encode_path = os.path.join(video_encode_feature_dir,"{}_encode_feature_frame_{}_{}.pt".format(half,video_encode_id*500,(video_encode_id+1)*500))
                if os.path.exists(video_encode_path):
                    frame_id = frame_stamp % 500
                    if video_encode_id == (frame_stamp//500) - 1:
                        if frame_stamp % 500 + 100 < 500:
                            video_encode_list.append(torch.tensor([1]))
                        else:
                            video_encode_list.append(torch.tensor([1]))
                    else:
                        video_encode_list.append(torch.tensor([1]))
                else:
                    continue
        torch.cuda.empty_cache()
        return video_encode_list



    #     #首先换到encode_feature 的路径
    # def _inner_training_loop(
    #     self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    # ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     if self.args.auto_find_batch_size:
    #         if self.state.train_batch_size != self._train_batch_size:
    #             from accelerate.utils import release_memory

    #             (self.model_wrapped,) = release_memory(self.model_wrapped)
    #             self.model_wrapped = self.model

    #             # Check for DeepSpeed *after* the intial pass and modify the config
    #             if self.is_deepspeed_enabled:
    #                 # Temporarily unset `self.args.train_batch_size`
    #                 original_bs = self.args.per_device_train_batch_size
    #                 self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
    #                 self.propagate_args_to_deepspeed(True)
    #                 self.args.per_device_train_batch_size = original_bs
    #         self.state.train_batch_size = self._train_batch_size
    #     logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     if self.is_fsdp_xla_v2_enabled:
    #         train_dataloader = tpu_spmd_dataloader(train_dataloader)

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

    #     len_dataloader = None
    #     num_train_tokens = None
    #     if has_length(train_dataloader):
    #         len_dataloader = len(train_dataloader)
    #         num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    #         num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    #         num_examples = self.num_examples(train_dataloader)
    #         if args.max_steps > 0:
    #             max_steps = args.max_steps
    #             num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
    #                 args.max_steps % num_update_steps_per_epoch > 0
    #             )
    #             # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
    #             # the best we can do.
    #             num_train_samples = args.max_steps * total_train_batch_size
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = (
    #                     self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #                 )
    #         else:
    #             max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    #             num_train_epochs = math.ceil(args.num_train_epochs)
    #             num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
    #             if args.include_tokens_per_second:
    #                 num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
    #     elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
    #         max_steps = args.max_steps
    #         # Setting a very large number of epochs so we go as many times as necessary over the iterator.
    #         num_train_epochs = sys.maxsize
    #         num_update_steps_per_epoch = max_steps
    #         num_examples = total_train_batch_size * args.max_steps
    #         num_train_samples = args.max_steps * total_train_batch_size
    #         if args.include_tokens_per_second:
    #             num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    #     else:
    #         raise ValueError(
    #             "args.max_steps must be set to a positive value if dataloader does not have a length, was"
    #             f" {args.max_steps}"
    #         )

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False

    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState()
    #     self.state.is_hyper_param_search = trial is not None
    #     self.state.train_batch_size = self._train_batch_size

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     if args.logging_steps is not None:
    #         if args.logging_steps < 1:
    #             self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
    #         else:
    #             self.state.logging_steps = args.logging_steps
    #     if args.eval_steps is not None:
    #         if args.eval_steps < 1:
    #             self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
    #         else:
    #             self.state.eval_steps = args.eval_steps
    #     if args.save_steps is not None:
    #         if args.save_steps < 1:
    #             self.state.save_steps = math.ceil(max_steps * args.save_steps)
    #         else:
    #             self.state.save_steps = args.save_steps

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         if args.gradient_checkpointing_kwargs is None:
    #             gradient_checkpointing_kwargs = {}
    #         else:
    #             gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

    #         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    #     model = self._wrap_model(self.model_wrapped)

    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = True if model is self.model else False

    #     if delay_optimizer_creation:
    #         if use_accelerator_prepare:
    #             self._fsdp_qlora_plugin_updates()
    #             self.model = self.accelerator.prepare(self.model)
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             if self.use_apex:
    #                 model = self.accelerator.prepare(self.model)
    #             else:
    #                 model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )

    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped

    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(
    #                 self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
    #             )
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    #     # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         self.compare_trainer_and_checkpoint_args(self.args, self.state)
    #         epochs_trained = self.state.global_step // num_update_steps_per_epoch
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )

    #     # Update the references
    #     self.callback_handler.model = self.model
    #     self.callback_handler.optimizer = self.optimizer
    #     self.callback_handler.lr_scheduler = self.lr_scheduler
    #     self.callback_handler.train_dataloader = train_dataloader
    #     if self.hp_name is not None and self._trial is not None:
    #         # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
    #         # parameter to Train when using DDP.
    #         self.state.trial_name = self.hp_name(self._trial)
    #     if trial is not None:
    #         assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
    #         self.state.trial_params = hp_params(assignments)
    #     else:
    #         self.state.trial_params = None
    #     # This should be the same if the state has been saved but in case the training arguments changed, it's safer
    #     # to set this after the load.
    #     self.state.max_steps = max_steps
    #     self.state.num_train_epochs = num_train_epochs
    #     self.state.is_local_process_zero = self.is_local_process_zero()
    #     self.state.is_world_process_zero = self.is_world_process_zero()

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0).to(args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #     grad_norm: Optional[float] = None

    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
    #     if not args.ignore_data_skip:
    #         for epoch in range(epochs_trained):
    #             sampler = get_dataloader_sampler(train_dataloader)
    #             sampler_kinds = [RandomSampler]
    #             if version.parse(accelerate_version) > version.parse("0.23.0"):
    #                 sampler_kinds.append(SeedableRandomSampler)
    #             is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
    #             if not is_random_sampler:
    #                 # We just need to begin an iteration to create the randomization of the sampler.
    #                 for _ in train_dataloader:
    #                     break
    #             else:
    #                 # Otherwise we need to call the whooooole sampler cause there is some random operation added
    #                 # AT THE VERY END!
    #                 sampler = sampler if sampler is not None else []
    #                 _ = list(sampler)
    #     total_batched_samples = 0
    #     for epoch in range(epochs_trained, num_train_epochs):
    #         epoch_iterator = train_dataloader
    #         if hasattr(epoch_iterator, "set_epoch"):
    #             epoch_iterator.set_epoch(epoch)

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_iterator)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if steps_trained_in_current_epoch > 0:
    #             epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         for step, inputs in enumerate(epoch_iterator):
    #             # import pdb
    #             # pdb.set_trace()
    #             input_device = inputs["labels"].device
    #             video_path = inputs["images"][0]
    #             timestamp = inputs["timestamp"]
    #             half = inputs["half"]
    #             # print(self.video_timestamp_to_video(video_path=video_path,timestamp=timestamp,half = half))
    #             # print(video_path,timestamp)
    #             # import pdb
    #             # pdb.set_trace()
    #             # continue
    #             # start_time = time.time()

    #             # # timestamp = 300
    #             # # video_path  = "/home/v-dingxin/blob/MatchTime/features_video/england_epl_2014-2015/2015-02-21_-_18-00_Chelsea_1_-_1_Burnley/1_224p.mkv"
    #             # video_list = self.video_timestamp_to_video_ddp(video_path=video_path,timestamp=timestamp,half = half)
    #             # if len(video_list)>0:
    #             #     video_encode_feature = torch.cat(video_list,1).to(input_device)
    #             #     del video_list
    #             # else:
    #             #     del video_list
    #             #     continue
    #             # # end_time = time.time()
    #             # # print("load data:",end_time-start_time)
                
    #             # # video_encode_feature = torch.cat(self.video_timestamp_to_video_test(video_path=video_path,timestamp=timestamp,half = half),1).to(input_device)
    #             # video_encode_feature = torch.cat(self.video_timestamp_to_video(video_path=video_path,timestamp=timestamp,half = half),1).to(input_device)
    #             # inputs["images"][0] = video_encode_feature
    #             total_batched_samples += 1

    #             if self.args.include_num_input_tokens_seen:
    #                 main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #                 if main_input_name not in inputs:
    #                     logger.warning(
    #                         "Tried to track the number of tokens seen, however the current model is "
    #                         "not configured properly to know what item is the input. To fix this, add "
    #                         "a `main_input_name` attribute to the model class you are using."
    #                     )
    #                 else:
    #                     input_device = inputs[main_input_name].device
    #                     self.state.num_input_tokens_seen += torch.sum(
    #                         self.accelerator.gather(
    #                             torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
    #                         )
    #                     ).item()
    #             if rng_to_sync:
    #                 self._load_rng_state(resume_from_checkpoint)
    #                 rng_to_sync = False

    #             # Skip past any already trained steps if resuming training
    #             if steps_trained_in_current_epoch > 0:
    #                 steps_trained_in_current_epoch -= 1
    #                 if steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.update(1)
    #                 if steps_trained_in_current_epoch == 0:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                 continue
    #             elif steps_trained_progress_bar is not None:
    #                 steps_trained_progress_bar.close()
    #                 steps_trained_progress_bar = None

    #             if step % args.gradient_accumulation_steps == 0:
    #                 self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #             with self.accelerator.accumulate(model):
    #                 tr_loss_step = self.training_step(model, inputs)

    #             if (
    #                 args.logging_nan_inf_filter
    #                 and not is_torch_xla_available()
    #                 and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #             ):
    #                 # if loss is nan or inf simply add the average of previous logged losses
    #                 tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #             else:
    #                 if tr_loss.device != tr_loss_step.device:
    #                     raise ValueError(
    #                         f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
    #                     )
    #                 tr_loss += tr_loss_step

    #             self.current_flos += float(self.floating_point_ops(inputs))

    #             is_last_step_and_steps_less_than_grad_acc = (
    #                 steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
    #             )

    #             if (
    #                 total_batched_samples % args.gradient_accumulation_steps == 0
    #                 or
    #                 # last step in epoch but step is always smaller than gradient_accumulation_steps
    #                 is_last_step_and_steps_less_than_grad_acc
    #             ):
    #                 # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
    #                 # in accelerate. So, explicitly enable sync gradients to True in that case.
    #                 if is_last_step_and_steps_less_than_grad_acc:
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                 # Gradient clipping
    #                 if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                     # deepspeed does its own clipping

    #                     if is_sagemaker_mp_enabled() and args.fp16:
    #                         _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    #                     elif self.use_apex:
    #                         # Revert to normal clipping otherwise, handling Apex or full precision
    #                         _grad_norm = nn.utils.clip_grad_norm_(
    #                             amp.master_params(self.optimizer),
    #                             args.max_grad_norm,
    #                         )
    #                     else:
    #                         _grad_norm = self.accelerator.clip_grad_norm_(
    #                             model.parameters(),
    #                             args.max_grad_norm,
    #                         )

    #                     if (
    #                         is_accelerate_available()
    #                         and self.accelerator.distributed_type == DistributedType.DEEPSPEED
    #                     ):
    #                         grad_norm = model.get_global_grad_norm()
    #                         # In some cases the grad norm may not return a float
    #                         if hasattr(grad_norm, "item"):
    #                             grad_norm = grad_norm.item()
    #                     else:
    #                         grad_norm = _grad_norm

    #                 # Optimizer step
    #                 self.optimizer.step()
    #                 optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
    #                 if optimizer_was_run:
    #                     # Delay optimizer scheduling until metrics are generated
    #                     if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                         self.lr_scheduler.step()

    #                 model.zero_grad()
    #                 self.state.global_step += 1
    #                 self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                 self.control = self.callback_handler.on_step_end(args, self.state, self.control)

    #                 self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    #             else:
    #                 self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                 # each step. Since we are breaking the loop early, we need to manually
    #                 # insert the mark_step here.
    #                 if is_torch_xla_available():
    #                     xm.mark_step()
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems to be not a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_xla_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sure the model has been saved by process 0.
    #         if is_torch_xla_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    #     train_loss = self._total_loss_scalar / effective_global_step

    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #         num_tokens=num_train_tokens,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()

    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)





    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # import pdb
        # pdb.set_trace()
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
    
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    # def compute_loss_score(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.

    #     Subclass and override for custom behavior.
    #     """
    #     # import pdb
    #     # pdb.set_trace()
    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None

    #     outputs = model(**inputs)
    #     # Save past state if it exists
    #     # TODO: this needs to be fixed and made cleaner later.
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     if labels is not None:
    #         unwrapped_model = unwrap_model(model)
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             loss = self.label_smoother(outputs, labels, shift_labels=True)
    #         else:
    #             loss = self.label_smoother(outputs, labels)
    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError(
    #                 "The model did not return a loss from the inputs, only the following keys: "
    #                 f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    #             )
    #         # We don't use .loss here since the model may return tuples instead of ModelOutput.
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss


#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666
RANK=0

# Training Arguments
GLOBAL_BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=2
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]
echo $LOCAL_BATCH_SIZE

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2_vllava
RUN_NAME=videollama2_stp_vllava
DATA_DIR=datasets
OUTP_DIR=work_dirs

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --deepspeed scripts/zero3.json \
    --version v1_mistral \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type stp_connector \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --data_path   ${DATA_DIR}/videollava_sft/videochatgpt_llavaimage_tune.json \
    --data_folder ${DATA_DIR}/videollava_sft/videochatgpt_tune \
    --pretrain_mm_mlp_adapter ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME}/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 99 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $RUN_NAME \

#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=16666
RANK=0

# Training Arguments
GLOBAL_BATCH_SIZE=32 #128
GRADIENT_ACCUMULATION_STEPS=2
LOCAL_BATCH_SIZE=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$GRADIENT_ACCUMULATION_STEPS)]
echo $LOCAL_BATCH_SIZE

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2_mamba
RUN_NAME=videollama2_mamba
DATA_DIR=/home/v-dingxin/blob/video_llm/datasets
OUTP_DIR=/home/v-dingxin/videollama2_plus-main/output
    # --pretrain_mm_mlp_adapter ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME}/mm_projector.bin \

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --output_dir ${OUTP_DIR}/${RUN_NAME}/finetune_${RUN_NAME} \
    --deepspeed scripts/zero3.json \
    --version v1_mistral \
    --tune_mm_mlp_adapter True \
    --vision_tower /home/v-dingxin/blob/clip-vit-large-patch14-336 \
    --freeze_backbone False \
    --mm_projector_type mamba \
    --model_name_or_path /home/v-dingxin/blob/VideoLLaMA2-7B \
    --data_path   ${DATA_DIR}/videollava_pt/valley_llavaimage_new.json \
    --data_folder ${DATA_DIR}/videollava_pt \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_frames 32 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
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
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --run_name $RUN_NAME \


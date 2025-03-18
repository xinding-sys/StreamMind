
# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per20/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per20/checkpoint-138/caption_per10/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.1 \
# --sample_type log

# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per20/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per20/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.2 \
# --sample_type log

# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per30/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per30/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.3 \
# --sample_type log

# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per40/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per40/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.4 \
# --sample_type log

# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.5 \
# --sample_type log


# python videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50_similarity/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50_similarity/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm \
# --sample_per 0.5 \
# --sample_type similarity


# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch15_ego4d_only_trainllm_1230_new_parallel_sample_resume_ck450_cls_2e6_focalloss_015_lastframe/checkpoint-480 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch15_ego4d_only_trainllm_1230_new_parallel_sample_resume_ck450_cls_2e6_focalloss_015_lastframe/checkpoint-480/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --ego4d_dataset \
# --data_type val \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2




python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-90 \
--caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-90/caption/ours_caption.csv \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type cls \
--sample_per 0.3 \
--sample_type log2


python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-100 \
--caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-100/caption/ours_caption.csv \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type cls \
--sample_per 0.3 \
--sample_type log2

python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-110 \
--caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-110/caption/ours_caption.csv \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type cls \
--sample_per 0.3 \
--sample_type log2
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_005/checkpoint-90 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_005/checkpoint-90/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2



# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_015/checkpoint-40 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_015/checkpoint-40/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2

# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_025/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_025/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2

# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_035/checkpoint-138 \
# --caption-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_ckpt138_clstrain_035/checkpoint-138/caption/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2


#ego4e_cls
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50/checkpoint-130 \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 16 \
# --ego4d_dataset \
# --data_type val \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2





# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_per50/checkpoint-130 \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 16 \
# --ego4d_dataset \
# --data_type val \
# --eval_type cls \
# --sample_per 0.3 \
# --sample_type log2






#soccer_cls
python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path /home/v-dingxin/videollm_paper_figure \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type cls

#soccer_llm
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_stream_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume/checkpoint-100 \
# --caption-path /home/v-dingxin/videollama2_plus-main/videollama2/eval2/ours_caption.csv \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --soccer_dataset \
# --data_type valid \
# --eval_type llm

#soccer_demo
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/video_score_stream_demo.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume_traincls_lastframe_001/checkpoint-90 \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --num-workers 16



python /home/v-dingxin/videollama2_plus-main/videollama2/eval/video_score_stream_demo.py \
--model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch15_ego4d_only_trainllm_1230_new_parallel_sample_resume_ck450_cls_2e6_focalloss_015_lastframe/checkpoint-480 \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--num-workers 16


#ego4d_lta_beam1
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_lta_parallel_new.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch3_soccer_only_trainllm_120_new_parallel_sample_resume/checkpoint-100 \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 1 \
# --ego4d_lta_dataset \
# --data_type val \
# --eval_type llm \
# --dataloader_num_workers 0



# #ego4d_lta_beam5_generation
# python /home/v-dingxin/videollama2_plus-main/videollama2/eval/inference_video_ego4d_lta_generate.py \
# --model-path /home/v-dingxin/blob/finetune_videollama2_mamba_batch1_stream_epoch15_ego4d_lta_only_trainllm_215_new_parallel_sample_resume/checkpoint-6050 \
# --model-name videollama_mamba-finetune \
# --eval-caption True \
# --cur_fps 2 \
# --soccer_dataset_train_llm \
# --num-workers 16 \
# --ego4d_lta_dataset \
# --data_type val \
# --eval_type llm \
# --dataloader_num_workers 16



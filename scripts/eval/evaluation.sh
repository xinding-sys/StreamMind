
#soccer_gate
python streammind/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path streammind_checkpoint \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type cls

#soccer_llm
python streammind/eval/inference_video_ego4d_stream_parallel_new.py \
--model-path streammind_checkpoint \
--caption-path streammind/eval2/ours_caption.csv \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--soccer_dataset_train_llm \
--num-workers 1 \
--soccer_dataset \
--data_type valid \
--eval_type llm

#soccer_demo
python streammind/eval/video_score_stream_demo.py \
--model-path streammind_checkpoint \
--model-name videollama_mamba-finetune \
--eval-caption True \
--cur_fps 2 \
--num-workers 16



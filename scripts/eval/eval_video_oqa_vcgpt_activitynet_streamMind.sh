set -x

EVAL_DATA_DIR=video_llm/eval
OUTPUT_DIR=eval_output
CKPT_NAME=finetune_videollama2_mamba
CKPT=finetune_streammind_live_llm_315_offline

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python3 videollama2/eval/inference_video_oqa_vcgpt.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/Test_Videos \
            --question-file ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/test_q.json \
            --answer-file ${EVAL_DATA_DIR}/Activitynet_Zero_Shot_QA/test_a.json \
            --output-file ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi


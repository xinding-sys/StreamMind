# set -x

EVAL_DATA_DIR=/home/v-dingxin/blob/video_llm/eval
OUTPUT_DIR=eval_output
CKPT_NAME=finetune_videollama2_mamba
CKPT=/home/v-dingxin/blob/finetune_streammind_live_llm_315_offline

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# divide data via the number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

output_file=${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/merge.json

if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # select the GPUs for the task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 videollama2/eval/inference_video_oqa_vcgpt_msvd.py \
            --model-path ${CKPT} \
            --video-folder ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/video \
            --gt_file ${EVAL_DATA_DIR}/MSVD_Zero_Shot_QA/test_qa.json \
            --output-file ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    #Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/${CHUNKS}_${IDX}.json >> "$output_file"
    done
fi




# AZURE_API_KEY=099e05b2850e4524bac53e57fd04ff68
# AZURE_API_ENDPOINT=https://mossai.openai.azure.com
# AZURE_API_DEPLOYNAME=gpt-35-turbo

# python3 videollama2/eval/eval_video_oqa_vcgpt_streammind_msvd.py \
#     --pred-path ${output_file} \
#     --output-dir ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/gpt \
#     --output-json ${OUTPUT_DIR}/MSVD_Zero_Shot_QA/answers/${CKPT_NAME}/results.json \
#     --api-key $AZURE_API_KEY \
#     --api-endpoint $AZURE_API_ENDPOINT \
#     --api-deployname $AZURE_API_DEPLOYNAME \
#     --num-tasks 4

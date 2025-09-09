CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model OpenGVLab/InternVL3-8B-Instruct \
    --model_type internvl3 \
    --infer_backend vllm \
    --agent_template hermes \
    --temperature 0.5 \
    --gpu_memory_utilization 0.8 \
    --max_model_len 4096 \
    --max_new_tokens 128 \
    --served_model_name InternVL3

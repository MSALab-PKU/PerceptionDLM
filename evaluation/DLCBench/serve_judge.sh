MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

vllm serve $MODEL_PATH \
    --served-model-name llama3.1-8b \
    --api-key sk-api \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.4 \
    --port 8007 \
    --host localhost

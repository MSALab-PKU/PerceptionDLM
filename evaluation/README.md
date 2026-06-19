# Evaluation Guide

This guide covers the evaluation of our models on broad Multimodal Benchmarks using a custom VLMEvalKit, as well as granular dense grounded evaluations.

## 1. Multimodal Benchmark Evaluation (VLMEvalKit)

We utilize a naturally modified [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) framework for comprehensive multimodal benchmark evaluations. 

### Setup

All evaluations require setting up `gpt-5.2` as the evaluator/judge. Please follow the VLMEvalKit guide to properly configure your API keys (e.g., setting up `OPENAI_API_KEY` or custom API bases).

```bash
export OPENAI_API_KEY="your_api_key_here"
# If necessary, export your base URL:
# export OPENAI_API_BASE="your_api_base_here"
```

### Running Benchmarks

We use `torchrun` to distribute the evaluation process across multiple GPUs. 

**Example command (Evaluating on 8 GPUs):**

```bash
cd VLMEvalKit
torchrun --nproc-per-node=8 --rdzv-backend=c10d  --rdzv-endpoint=localhost:13501 run.py \
    --data MMVP BLINK RealWorldQA CV-Bench-2D HallusionBench VStarBench \
    --model PerceptionDLM-Base \
    --verbose
```

### Post-processing for Document & Chart Benchmarks

For **ChartQA**, **DocVQA**, and **InfoVQA**, we perform an advanced and stricter post-processing evaluation to guarantee accurate metric reflection. We employ a local `Qwen/Qwen3-8B` deployed via vLLM as the strict judge.

**Step 1: Launch the vLLM server**

```bash
# Set NUM_GPUS according to your environment
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 8
```

**Step 2: Run the Judge Evaluator**

After VLMEvalKit generates the `.xlsx` prediction outputs, modify the `input_xlsx` and `output_xlsx` paths within `evaluation/judge.py` to point to the freshly generated logs, and run:

```bash
cd ../ # Return to project root
python evaluation/judge.py
```

---

## 2. Dense Grounded Evaluation (ParaDLC-Bench)

Evaluation consists of a two-step process: running inference to generate captions, followed by calling a judge model (e.g., GPT) for evaluation. 

### 1. Inference

Run the inference script to generate mask captions:

```bash
python evaluation/ParaDLC-Bench/infer_mask_captions_paradlc.py \
    --model-path /path/to/PerceptionDLM \
    --image-root annotations/images \
    --anno-json annotations/annotations.json \
    --qa-json annotations/qa.json \
    --prompt "Describe the masked region in detail." \
    --gen-length 32 --steps 32 --temperature 0.0 --top-p 1.0 \
    --cache-name-override paradlc_outputs
```

### 2. Judge Evaluation

Run the GPT evaluation script on the generated outputs:

```bash
python evaluation/ParaDLC-Bench/eval_gpt_with_image.py \
    --pred parallel_model_outputs_cache/paradlc_outputs.json \
    --qa annotations/qa.json \
    --class-names annotations/class_names.json \
    --anno-json annotations/annotations.json
```

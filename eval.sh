#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1
# export PYTHON_START_METHOD=spawn
MODEL_PATH=./
OUTPUT_DIR=./
mkdir -p "$OUTPUT_DIR"
echo "Evaluating $MODEL_PATH"

# nohup sh eval.sh > eval.log 2>&1 &

python3 eval.py \
  --model_name="$MODEL_PATH" \
  --datasets="./test_data/MATH" \
  --split="test" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=1000 \
  --max_tokens=4096 \
  --num_gpus=2 \
  --temperature=0.6 \
  --top_p=0.95 \
  --num_generation=256

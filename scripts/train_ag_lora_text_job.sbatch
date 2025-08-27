#!/bin/bash
#SBATCH -J ag_lora_text_training
#SBATCH -p ba100
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/ag_lora_text_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate vqa-research

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs"
cd "$WORKDIR"

# LoRA fine-tuning without video (faster, single GPU)
python scripts/train_ag_lora.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl submissions/ag_train.jsonl \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir outputs/ag-qwen-lora-text-10answers \
  --num_answers 10 \
  --lr 3e-4 --batch_size 2 --grad_accum 8 --epochs 5 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --warmup_steps 100 --save_steps 500 --eval_steps 500 \
  --logging_steps 50 --max_length 2048

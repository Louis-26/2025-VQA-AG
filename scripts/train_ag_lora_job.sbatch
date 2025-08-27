#!/bin/bash
#SBATCH -J ag_lora_training
#SBATCH -p ba100
#SBATCH --gpus=2
#SBATCH -t 24:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/ag_lora_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework

# Initialize conda environment
export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
source activate vqa-research

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs"
cd "$WORKDIR"

# LoRA fine-tuning with video (DDP across 2 GPUs)
export MASTER_PORT=${MASTER_PORT:-29504}

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT scripts/train_ag_lora.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl submissions/ag_train.jsonl \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir outputs/ag-qwen-lora-video-10answers \
  --use_video --num_frames 32 --num_answers 10 \
  --lr 2e-4 --batch_size 1 --grad_accum 8 --epochs 5 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --warmup_steps 100 --save_steps 500 --eval_steps 500 \
  --logging_steps 50 --max_length 2048

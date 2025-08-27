#!/bin/bash
#SBATCH -J ag_full_training
#SBATCH -p ba100
#SBATCH --gpus=4
#SBATCH -t 48:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/ag_full_%j.out

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

# Full fine-tuning with video (DDP across 4 GPUs)
export MASTER_PORT=${MASTER_PORT:-29503}

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT scripts/train_ag_full.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl submissions/ag_train.jsonl \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir outputs/ag-qwen-full-video-10answers \
  --use_video --num_frames 32 --num_answers 10 \
  --lr 1e-5 --batch_size 1 --grad_accum 4 --epochs 3 \
  --warmup_steps 100 --save_steps 500 --eval_steps 500 \
  --logging_steps 50 --max_length 2048

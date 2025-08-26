#!/bin/bash
#SBATCH -J reranker_lora_video
#SBATCH -p ba100
#SBATCH --gpus=1
#SBATCH -t 1:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/reranker_lora_video_%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework

# Initialize conda for this script
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate vqa-research

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs"
cd "$WORKDIR"

python scripts/train_reranker.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train.jsonl \
  --output_dir /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/outputs/reranker-lora-video-w4e3 \
  --lr 5e-5 --batch_size 1 --epochs 3 \
  --w1 4.0 --w2 2.0 --w3 1.2 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --use_video --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos --num_frames 16



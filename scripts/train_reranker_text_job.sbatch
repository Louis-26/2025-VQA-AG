#!/bin/bash
#SBATCH -J reranker_lora_text_lr1e-5
#SBATCH -p ba100
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/reranker_lora_text_%j.out

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
  --train_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train_teacher.gold_plus_gen.train.jsonl \
  --val_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train_teacher.gold_plus_gen.dev.jsonl \
  --output_dir /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/outputs/reranker-lora-text-lr1e-5-bs8-ga4-e5 \
  --lr 1e-5 --batch_size 20 --grad_accum 4 --epochs 10 \
  --w1 6.0 --w2 2.0 --w3 1.2 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05



#!/bin/bash
#SBATCH -J reranker_lora_asr_epoch
#SBATCH -p ba100
#SBATCH --gpus=4
#SBATCH -t 24:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/reranker_lora_asr_%j.out

set -eo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework

# Initialize conda for this script
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate vqa-research

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs" "/brtx/603-nvme2/yweng13/checkpoints"
cd "$WORKDIR"

# Use all available GPUs on the node (fallback to 4)
NPROC=${SLURM_GPUS_ON_NODE:-4}

# Launch DDP
torchrun --standalone --nproc_per_node=${NPROC} scripts/train_reranker.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_from_xlsx.train.jsonl \
  --val_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_from_xlsx.dev.jsonl \
  --output_dir /brtx/603-nvme2/yweng13/checkpoints/reranker-lora-asr-epoch \
  --lr 2e-6 --batch_size 1 --epochs 10 \
  --w1 3.0 --w2 1.5 --w3 1.2 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05



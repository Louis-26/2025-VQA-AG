#!/bin/bash
#SBATCH -J ag_lora_video_epoch
#SBATCH -p ba100
#SBATCH --gpus=3
#SBATCH -t 10:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/ag_lora_video_epoch_%j.out

set -eo pipefail  # Remove 'u' flag to allow unbound variables during conda setup

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR=/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework

# Initialize conda (allow unbound variables for conda scripts)
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vqa-research
set -u  # Re-enable unbound variable checking after conda activation

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs"
cd "$WORKDIR"

# Install matplotlib if not available
pip install matplotlib seaborn || echo "matplotlib installation failed, plotting may not work"

# Generate timestamp for unique output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/ag-qwen-lora-video-epoch-${TIMESTAMP}"

echo "Starting LoRA training with video + epoch checkpoints..."
echo "Output directory: $OUTPUT_DIR"
echo "Training will save checkpoints after every epoch and plot loss curves"

# LoRA fine-tuning with video (DDP across 2 GPUs) with epoch checkpoints
export MASTER_PORT=${MASTER_PORT:-29504}

torchrun --nproc_per_node=2 --master_port=$MASTER_PORT scripts/train_ag_lora.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl submissions/ag_train.jsonl \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir "$OUTPUT_DIR" \
  --use_video --num_frames 32 --num_answers 10 \
  --lr 2e-4 --batch_size 1 --grad_accum 8 --epochs 10 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --warmup_steps 100 --logging_steps 25 \
  --save_every_epoch \
  --plot_loss \
  --max_length 2048

echo "Training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "Loss plot saved as: ${OUTPUT_DIR}/training_loss_plot.png"

# List the created checkpoints
echo -e "\nCreated checkpoints:"
ls -la "$OUTPUT_DIR"/checkpoint-* 2>/dev/null || echo "No checkpoints found"

# Show final loss plot location
if [ -f "${OUTPUT_DIR}/training_loss_plot.png" ]; then
    echo -e "\n📊 Training loss plot available at:"
    echo "   ${OUTPUT_DIR}/training_loss_plot.png"
    echo -e "\n📈 Loss data available at:"
    echo "   ${OUTPUT_DIR}/training_loss_data.csv"
fi


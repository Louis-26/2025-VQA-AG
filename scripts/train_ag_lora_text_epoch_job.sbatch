#!/bin/bash
#SBATCH -J ag_lora_text_epoch
#SBATCH -p brtx6
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -o /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/logs/ag_lora_text_epoch_%j.out

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
OUTPUT_DIR="outputs/ag-qwen-lora-text-epoch-${TIMESTAMP}"

echo "Starting LoRA training with epoch checkpoints..."
echo "Output directory: $OUTPUT_DIR"
echo "Training will save checkpoints after every epoch and plot loss curves"

# LoRA fine-tuning without video (faster, single GPU) with epoch checkpoints
python scripts/train_ag_lora.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl submissions/ag_train.jsonl \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir "$OUTPUT_DIR" \
  --num_answers 10 \
  --lr 3e-4 --batch_size 2 --grad_accum 8 --epochs 5 \
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


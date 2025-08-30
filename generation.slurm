#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --output=./log/generation.out
#SBATCH --error=./log/generation.err
#SBATCH --partition=ba100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00

source ~/.bashrc
conda init
conda activate qwen-omni

cd /home/dzhang98/code/2025-VQA-AG


python run_zeroshot_vqa.py \
  --model_config qwen_omni_vllm \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --num_answers 10 \
  --output submissions/qwen_omni_vllm.csv 

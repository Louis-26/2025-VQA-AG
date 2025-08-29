#!/bin/bash
#SBATCH --job-name=generation
#SBATCH --output=./log/generation.out
#SBATCH --error=./log/generation.err
#SBATCH --partition=ba100
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00

source ~/.bashrc
conda init
conda activate video-r1

cd /home/dzhang98/code/2025-VQA-AG

num=5

python run_zeroshot_vqa.py \
  --model_config qwen_vl_chat_vllm \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --num_answers 10 \
  --output submissions/qwen_candidates_Plora_${num}.csv \
  --prompt_type ${num}

python evaluation/evaluate_ag_results.py \
  --pred_file submissions/qwen_candidates_Plora_${num}.csv \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --normalize \
  --bertscore_model distilbert-base-uncased

  


rank_num=Plora_5
python scripts/rerank_with_sts.py \
  --pred_file submissions/qwen_candidates_${rank_num}.csv \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --enable_rerank \
  --normalize \
  --reranked_out submissions/qwen_candidates_${rank_num}_reranked.csv
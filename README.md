# TRECVID 2025 VQA Research Framework

Framework for the TRECVID 2025 Video Question Answering challenge with:
- Zero‑shot Answer Generation (Qwen2.5‑VL, optional vLLM)
- Pointer‑List Reranker (Qwen2.5‑VL + LoRA)
- LLaVA‑Critic based reranking
- Evaluation utilities

Official task: https://www-nlpir.nist.gov/projects/tv2025/vqa.html

## Implemented components
- Answer generation pipelines
  - `run_zeroshot_vqa.py`: Qwen2.5‑VL (transformers) or vLLM (`model_config=qwen_vl_chat|qwen_vl_chat_vllm`).
- Pointer‑List Reranker (text‑only by default)
  - Dataset builder: `scripts/build_reranker_dataset.py`
    - FIX: Candidate enumeration is now consistent between prompt and target. The builder uses a single enumeration for both, guaranteeing `<R1>` points to the gold candidate ID in the prompt.
  - Trainer: `scripts/train_reranker.py` (Qwen2.5‑VL + LoRA, rank‑weighted CE, pointer masking)
  - Eval: `scripts/eval_reranker.py` (batched constrained decoding, Top‑1/NDCG@10)
- LLaVA‑Critic reranker: `scripts/run_rerank_with_critic.py` (scores and reranks generator candidates from frames + question)

## Quickstart

### 1) Build reranker dataset from generator candidates
Input: CSV/XLSX with columns `Q_ID,Video_ID,Rank,Answer` and JSONs with `question`, `correct_answer`, `incorrect_answers`.
```bash
python scripts/build_reranker_dataset.py \
  --candidates_csv "/path/to/qwen_candidates.xlsx" \
  --json_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --teacher_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_jsonl submissions/reranker_train_teacher.jsonl
```

Split into train/dev:
```bash
python scripts/split_jsonl.py \
  --input_jsonl submissions/reranker_train_teacher.jsonl \
  --out_train_jsonl submissions/reranker_train_teacher.train.jsonl \
  --out_dev_jsonl submissions/reranker_train_teacher.dev.jsonl \
  --dev_ratio 0.1 --seed 42
```

### 2) Train reranker (text‑only, LoRA)
```bash
python scripts/train_reranker.py \
  --train_jsonl submissions/reranker_train_teacher.train.jsonl \
  --val_jsonl submissions/reranker_train_teacher.dev.jsonl \
  --output_dir outputs/reranker_lora \
  --use_lora --epochs 1 --batch_size 1 --lr 2e-6 --w1 3.0 --w2 1.5 --w3 1.2
```

Optional video mode (heavier): add `--use_video --videos_dir /path/to/videos --num_frames 64`.

### 3) Evaluate reranker (constrained decoding)
```bash
python scripts/eval_reranker.py \
  --jsonl submissions/reranker_train_teacher.dev.jsonl \
  --adapters outputs/reranker_lora \
  --max_examples 200 --batch_size 16
```
Outputs: Top‑1, NDCG@10, validity (no repeats).

### 4) Zero‑shot candidate generation (optional)
```bash
python run_zeroshot_vqa.py \
  --model_config qwen_vl_chat_vllm \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --num_answers 10 \
  --output submissions/qwen_candidates_3.csv
```

You can evaluate the result:
```bash
python evaluation/evaluate_ag_results.py \
  --pred_file submissions/qwen_candidates_2.csv \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --normalize
```

### 5) Rerank with LLaVA‑Critic (optional)
```bash
python -m scripts.run_rerank_with_critic \
  --candidates_csv submissions/qwen_candidates.csv \
  --json_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_csv submissions/qwen_candidates.reranked.csv \
  --max_images 8 --max_decode_frames 256
```
### 6) Generate ASR transcripts 
```
python scripts/generate_transcript.py \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_dir ./submissions
```
## How the reranker works
- Prompt contains: Question, optional ASR, and a “Candidates:” block with `<CAND_i>: text` lines.
- Target is a pointer list: `<R1>=<CAND_j>`, `<R2>=<CAND_k>`, … `<ENDLIST>`.
- Training loss: rank‑weighted CE with vocabulary constraints at pointer positions (only remaining `<CAND_*>` or `<ENDLIST>` are allowed). Early ranks receive higher weights.
- Teacher ordering (optional): tail ranks ordered by sentence‑embedding similarity to the gold.

## Notes
- XLSX input requires `openpyxl`.
- Some AV1 videos may fail to decode; such samples are skipped with warnings.
- If your git ignore excludes `submissions/*.csv`, force‑add as needed.

## Repo structure (key files)
```
├── scripts/
│   ├── build_reranker_dataset.py   # Build JSONL from generator candidates + GT
│   ├── train_reranker.py           # LoRA training (text‑only by default; optional video)
│   ├── eval_reranker.py            # Batched constrained decoding eval
│   └── split_jsonl.py              # Train/dev split utility
├── src/reranker/
│   ├── prompter.py                 # Candidate enumeration, prompt/target formatting
│   ├── losses.py, masking.py       # Rank‑weighted CE, pointer masking
│   ├── decoding.py                 # Greedy/batched pointer decoding
│   └── tokens.py                   # Special token registration
├── src/ag_task/
│   ├── vqa_model_vllm.py           # vLLM + transformers Qwen2.5‑VL inference
│   └── critic_reranker.py          # LLaVA‑Critic wrapper
```

## License
MIT. See `LICENSE`.

## Qwen‑2.5‑VL Pointer‑List Reranker (LoRA)

This module fine‑tunes Qwen‑2.5‑VL to emit a ranked list of candidate IDs using a pointer format:

Reasoning: <one line>
Final list:
<R1>=<CAND_j>
<R2>=<CAND_k>
...
<ENDLIST>

Design:
- Special tokens: `<R1>…<R10>`, `<CAND_1>…<CAND_16>`, `<ENDLIST>` are added to the tokenizer.
- Candidate IDs are randomized per example; list length fixed to R=10 (early stop allowed if M<10).
- Training uses rank‑weighted CE with training‑time vocab constraints (only remaining `<CAND_*>` allowed at each pointer).
- Inputs are text‑only for now: Question, (optional ASR), and enumerated candidates; video can be added later.

### 1) Build training JSONL

Requires:
- Generator outputs CSV: `Q_ID,Video_ID,Rank,Answer,Time (sec)` (e.g., `qwen_candidates(in).csv`)
- GT JSONs at `/brtx/603-nvme1/yweng13/VQA/train_json_files` with keys: `question`, `correct_answer`, `incorrect_answers` (list)

Command:
```bash
python scripts/build_reranker_dataset.py \
  --candidates_csv \
  "/brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/qwen_candidates(in).csv" \
  --json_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --out_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train.jsonl
```

Output JSONL fields: `prompt`, `target`, plus metadata (`Q_ID`, `Video_ID`, `candidates`, `gold`, `wrong`).

### 2) Train (LoRA, masked rank‑weighted CE)

```bash
python scripts/train_reranker.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train.jsonl \
  --output_dir /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/outputs/reranker-lora \
  --lr 1e-4 --batch_size 1 --epochs 3 \
  --w1 4.0 --w2 2.0 --w3 1.2 \
  --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

Notes:
- Gradient checkpointing is enabled; `use_cache=False` auto‑set.
- For OOM risk, set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### 3) Evaluate (model‑in‑loop constrained decoding)

```bash
python scripts/eval_reranker.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapters /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/outputs/reranker-lora \
  --jsonl /brtx/603-nvme1/yweng13/trecvid/my-vqa-research-framework/submissions/reranker_train.jsonl \
  --max_examples 200
```

Reports: `Top-1`, `NDCG@10`, and `validity` (no repeats). For a fast no‑model scorer, add later.

### 4) Serve with vLLM

If using LoRA, merge adapters into full weights, then point vLLM to the merged folder.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-VL-7B-Instruct"
adapters = "/brtx/.../outputs/reranker-lora"
merge_out = "/brtx/.../outputs/reranker-merged"

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapters)
model = model.merge_and_unload()
model.save_pretrained(merge_out)
tok.save_pretrained(merge_out)
```

Then update your vLLM config to `model_name=merge_out`.

### File Map
- Tokens/formatting: `src/reranker/constants.py`, `src/reranker/tokens.py`, `src/reranker/prompter.py`
- Dataset: `src/reranker/build_dataset.py`, `scripts/build_reranker_dataset.py`
- Loss/masking: `src/reranker/losses.py` (masked pointer CE + rank weights)
- Training: `scripts/train_reranker.py` (HF + LoRA)
- Decoding/Eval: `src/reranker/decoding.py`, `scripts/eval_reranker.py`



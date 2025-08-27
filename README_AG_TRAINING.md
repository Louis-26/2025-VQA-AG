# Answer Generation (AG) Model Training

This document describes the complete training pipeline for fine-tuning Qwen2.5-VL-7B-Instruct to generate 10 diverse answers and maximize BERTScore with ground truth.

## 🚀 Quick Start

**For immediate testing:**
```bash
# 1. Build dataset
python scripts/build_ag_dataset.py \
  --json_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --output_jsonl submissions/ag_train.jsonl \
  --asr_file transcripts.json

# 2. Start text-only training (fastest)
sbatch scripts/train_ag_lora_text_job.sbatch
```

## 📁 Key Files for Code Review

### Core Implementation Files
- **`src/ag_task/dataset.py`** - Dataset loading and preprocessing logic
- **`src/ag_task/losses.py`** - Custom BERTScore maximization loss functions
- **`src/ag_task/collators.py`** - Data collation for multimodal inputs
- **`scripts/train_ag_lora.py`** - Main LoRA training script (recommended)
- **`scripts/train_ag_full.py`** - Full fine-tuning script (high memory)

### Dataset Building
- **`scripts/build_ag_dataset.py`** - Converts GT JSONs + ASR to training format
- **`scripts/build_ag_dataset.py:load_asr_data()`** - Handles multiple ASR formats

### Batch Job Scripts
- **`scripts/train_ag_lora_text_job.sbatch`** - 1-GPU text-only training
- **`scripts/train_ag_lora_job.sbatch`** - 2-GPU video+text training  
- **`scripts/train_ag_full_job.sbatch`** - 4-GPU full fine-tuning

## 🧠 Training Objective

**Goal:** Train Qwen2.5-VL to generate 10 answers (a1, a2, ..., a10) and maximize:
```
max(BERTScore(a1, GT), BERTScore(a2, GT), ..., BERTScore(a10, GT))
```

**Implementation:** See `src/ag_task/losses.py:BERTScoreMaxLoss`

## 📊 Dataset Format

### Input: Ground Truth JSON Files
Location: `/brtx/603-nvme1/yweng13/VQA/train_json_files/*.json`

Each file (`{Video_ID}.json`) contains:
```json
{
  "Q_ID": "q_B24dV_uori4",
  "question": "What causes shampoo to lather best when washing hair?",
  "correct_answer": "Washing your hair twice gives you cleaner hair and better lather."
}
```

### Input: ASR Transcripts
File: `transcripts.json` (array format)
```json
[
  {
    "Q_ID": "q_B24dV_uori4",
    "Video_ID": "B24dV_uori4", 
    "Question": "What causes shampoo to lather best?",
    "correct_answer": "Washing your hair twice...",
    "transcript": "He's washing his hair with shampoo..."
  }
]
```

### Output: Training Dataset
File: `submissions/ag_train.jsonl` (501 examples)
```json
{"Q_ID": "q_B24dV_uori4", "Video_ID": "B24dV_uori4", "question": "What causes shampoo to lather best when washing hair?", "asr_transcript": "", "ground_truth": "Washing your hair twice gives you cleaner hair and better lather."}
```

## 🛠 Training Scripts Deep Dive

### 1. LoRA Training (`scripts/train_ag_lora.py`)

**Key Components:**
- **Line 45-65:** LoRA configuration (rank=32, alpha=64)
- **Line 89-120:** Custom data collator selection (video vs text-only)
- **Line 156-180:** BERTScore loss integration
- **Line 220-250:** Hugging Face Trainer setup with custom loss

**Critical Parameters:**
```python
# LoRA config
lora_config = LoraConfig(
    r=32,                    # Rank (higher = more capacity)
    lora_alpha=64,           # Scaling factor (2x rank)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1
)

# Training args
learning_rate=2e-4          # Higher than full fine-tuning
per_device_train_batch_size=1
gradient_accumulation_steps=8
num_train_epochs=5
```

### 2. Full Fine-tuning (`scripts/train_ag_full.py`) 

**Key Differences from LoRA:**
- **Line 78:** No PEFT wrapping - trains all parameters
- **Line 145:** Lower learning rate (1e-5 vs 2e-4)
- **Line 200:** Requires more aggressive gradient accumulation

### 3. Data Collators (`src/ag_task/collators.py`)

**AGVideoCollator (Lines 15-160):**
- Loads video frames from `file://` URIs
- Samples 32 frames uniformly
- Formats prompt for 10-answer generation
- Creates labels for teacher forcing

**AGTextCollator (Lines 162-309):**
- Text-only version (no video processing)
- Same prompt formatting
- Much faster for debugging

**Key Method:** `format_ag_prompt()` (Line 25) - Creates the instruction template

## 📈 Loss Function (`src/ag_task/losses.py`)

### BERTScoreMaxLoss (Lines 15-85)
```python
def forward(self, logits, labels, input_ids=None):
    # 1. Generate text from logits
    generated_text = self.extract_answers(logits, input_ids)
    
    # 2. Parse 10 answers from generated text  
    answers = self.parse_answers(generated_text)
    
    # 3. Calculate BERTScore vs ground truth
    scores = self.bertscore(answers, [ground_truth] * len(answers))
    
    # 4. Return negative max score (for minimization)
    return -torch.max(scores['f1'])
```

**Key Challenge:** Backpropagation through text generation (experimental)

### SimpleBERTScoreMaxLoss (Lines 87-120)
Simplified version for stability testing.

## 🚀 Training Commands

### Option 1: Text-Only LoRA (Recommended First)
```bash
# Submit job
sbatch scripts/train_ag_lora_text_job.sbatch

# Monitor
tail -f outputs/ag-qwen-lora-text-*/training.log
```

**Settings:**
- 1 GPU, 2 hours estimated
- Batch size: 2, Grad accum: 8 (effective batch = 16)
- Learning rate: 3e-4
- No video processing

### Option 2: Video + Text LoRA  
```bash
# Submit job
sbatch scripts/train_ag_lora_job.sbatch

# Check GPU usage
squeue -u $USER
```

**Settings:**
- 2 GPUs, 4-6 hours estimated  
- Batch size: 1, Grad accum: 8 (effective batch = 16)
- 32 video frames per sample
- Learning rate: 2e-4

### Option 3: Full Fine-tuning (Resource Intensive)
```bash
# Submit job (requires 4 A100s)
sbatch scripts/train_ag_full_job.sbatch
```

**Settings:**
- 4 GPUs, 8+ hours estimated
- Much higher memory usage
- Learning rate: 1e-5 (lower than LoRA)

## 🔍 Monitoring Training

### Key Metrics to Watch
```bash
# Training loss (should decrease)
grep "train_loss" outputs/ag-qwen-*/training.log

# GPU memory usage  
nvidia-smi

# Generated samples (check format)
tail outputs/ag-qwen-*/checkpoint-*/generation_samples.txt
```

### Expected Output Format
```
1. A person is walking down the street
2. Someone is taking a stroll  
3. An individual is moving on foot
4. A pedestrian is walking
5. A person is going for a walk
6. Someone is strolling along
7. A walker is on the sidewalk
8. A person is taking steps
9. An individual is ambulatory  
10. Someone is proceeding on foot
```

## 🐛 Common Issues & Solutions

### 1. CUDA Out of Memory
```bash
# Reduce batch size in sbatch script
--per_device_train_batch_size 1
--gradient_accumulation_steps 16  # Increase to maintain effective batch size
```

### 2. Bad Generation Format
**Symptom:** Model generates non-numbered lists or stops early

**Debug:** Check `src/ag_task/collators.py:format_ag_prompt()` - ensure proper instruction format

### 3. BERTScore Issues  
**Symptom:** `ImportError: sentence-transformers`

**Fix:** 
```bash
pip install sentence-transformers==2.2.2
pip install bert-score==0.3.13
```

### 4. Video Loading Errors
**Symptom:** `FileNotFoundError` for video files

**Fix:** Ensure video paths in dataset match actual file locations, or use text-only mode first

## 📁 Output Structure

```
outputs/ag-qwen-lora-text-[timestamp]/
├── checkpoint-500/           # Saved every 500 steps
│   ├── adapter_model.bin     # LoRA weights  
│   ├── adapter_config.json   # LoRA configuration
│   └── training_args.bin     # Training arguments
├── training.log              # Full training log
├── training_args.json        # Final training config
└── generation_samples.txt    # Sample outputs during training
```

## 🧪 Testing Trained Model

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Load base model
base_model = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(base_model)

# Load your trained adapter
model = Qwen2VLForConditionalGeneration.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, "outputs/ag-qwen-lora-text-[timestamp]/checkpoint-1000")

# Test generation
prompt = """Question: What do you see in this video?
ASR Transcript: A person is walking down the street.

Please provide exactly 10 different possible answers to this question based on the video content and transcript.
Format your response as a numbered list from 1 to 10:

"""

inputs = processor(text=prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## 📋 Code Review Checklist

For reviewers, please check:

- [ ] **Dataset logic** (`scripts/build_ag_dataset.py`) handles ASR format correctly
- [ ] **Loss function** (`src/ag_task/losses.py`) implements BERTScore maximization 
- [ ] **Collators** (`src/ag_task/collators.py`) format prompts to generate exactly 10 answers
- [ ] **Training scripts** handle both LoRA and full fine-tuning correctly
- [ ] **Batch scripts** have reasonable resource allocation
- [ ] **Video processing** efficiently samples frames without memory leaks
- [ ] **Error handling** for missing videos/transcripts
- [ ] **Checkpointing** saves adapters correctly for LoRA

## 🚨 Known Limitations

1. **BERTScore Training:** Experimental - may be unstable for large batches
2. **Video Memory:** Loading 32 frames per sample is memory-intensive  
3. **Generation Quality:** May need prompt engineering for consistent formatting
4. **Evaluation:** No automatic BERTScore evaluation during training yet

## 📞 Support

For issues, check:
1. Training logs in `outputs/ag-qwen-*/training.log`
2. Slurm logs: `cat slurm-[job_id].out`
3. GPU usage: `nvidia-smi` 
4. Data format: `head submissions/ag_train.jsonl`
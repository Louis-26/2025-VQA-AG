# TRECVID 2025 VQA Research Framework

A flexible and research-oriented framework for the **TRECVID 2025 Video Question Answering Challenge**, designed for experimenting with advanced multimodal models, grounding techniques, and synthetic data augmentation.

**Official Task Details**: https://www-nlpir.nist.gov/projects/tv2025/vqa.html

## What's New (Zero-shot Qwen2.5‑VL pipeline)

- **Qwen 2.5‑VL 7B Instruct integration** for zero-shot video VQA
  - Config: `Qwen/Qwen2.5-VL-7B-Instruct` in `src/ag_task/model_configs.py`
  - New specialized loader: `QwenVQAModel` in `src/ag_task/vqa_model.py` (native video handling via processor)
- **Robust data loading for your JSON format**: `src/ag_task/json_data_loader.py`
- **End-to-end inference script**: `run_zeroshot_vqa.py`
  - Skips missing/corrupted videos gracefully
  - Generates 16 candidates per item and de-duplicates to top‑10
- **Critic reranking with LLaVA‑Critic (separate env)**: `scripts/run_rerank_with_critic.py`
  - Uses sampled frames from videos and judges cand. answers; requires `lmms-lab/llava-critic-7b`
  - Loads with `attn_implementation="sdpa"` to avoid FlashAttention2 deps
- **Evaluation**: `evaluation/evaluate_ag_results.py`
  - Reports ROUGE‑L, METEOR, BERTScore, STS; optional text normalization

### Minimal end-to-end usage

```bash
# 1.a) Generate candidates with Qwen 2.5‑VL (zero-shot) 
python run_zeroshot_vqa.py \
  --model_config qwen_vl_chat \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --num_answers 16 \
  --max_videos -1 \
  --output submissions/qwen_candidates.csv

# 1.b) Generate candidates with Qwen 2.5‑VL vllm (zero-shot) 
python run_zeroshot_vqa.py \
  --model_config qwen_vl_chat_vllm \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --num_answers 10 \
  --max_videos -1 \
  --output submissions/qwen_candidates.csv

# 2) Rerank with LLaVA‑Critic (run in separate vqa-critic env)
python -m scripts.run_rerank_with_critic \
  --candidates_csv submissions/qwen_candidates_small.csv \
  --json_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_train_videos \
  --output_csv submissions/qwen_candidates.reranked.csv \
  --max_images 8 --max_decode_frames 256

# 3) Evaluate
python evaluation/evaluate_ag_results.py \
  --pred_file submissions/qwen_candidates.reranked.csv \
  --json_files_dir /brtx/603-nvme1/yweng13/VQA/train_json_files \
  --normalize
```

### LLaVA‑Critic environment tip

- Install LLaVA‑NeXT and pins:
  - `pip install --no-cache-dir git+https://github.com/LLaVA-VL/LLaVA-NeXT.git@main`
  - `pip install "transformers==4.43.3" "huggingface_hub==0.24.2" "accelerate==0.33.0"`
  - `pip install av einops open-clip-torch timm`
- The reranker loads with `attn_implementation="sdpa"` by default.

### Notes

- Missing videos are skipped with a warning; run will continue.
- `.gitignore` ignores `submissions/*.csv`, so force-add when committing results.

## 🚀 Key Features

- **🔄 Flexible Model Architecture**: Easy swapping between different VQA models (ViT+T5, LLaVA, InstructBLIP, etc.)
- **🎯 Grounding & Alignment**: Retrieval-augmented generation and cross-verification to reduce hallucination
- **🤖 Synthetic Data Generation**: Template-based, LLM-based, and back-translation approaches for data augmentation
- **⚙️ Configuration-Driven**: JSON-based model and pipeline configurations
- **📊 Research-Ready**: Built for experimentation with state-of-the-art techniques

## 📋 Challenge Tasks

### 1. Answer Generation (AG) Task
Generate up to 10 ranked natural language answers for video questions.
- **Input**: Video (~30 seconds) + Question
- **Output**: Ranked answers with confidence scores and timing
- **Evaluation**: NDCG, STS, METEOR, BERTScore

### 2. Multiple Choice (MC) Task *(TODO)*
Rank 4 provided answer options for video questions.

## 🛠️ Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd trec-project-template

# Set up environment with uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Basic Usage
```bash
# List available model configurations
python run_ag_task.py --list_configs

# Run Answer Generation with baseline model
python run_ag_task.py \
    --topics_file data/sample_topics.csv \
    --videos_dir data/videos/ \
    --output_file submissions/team_ag_run1.csv \
    --model_config baseline_encoder_decoder

# Run with grounding enabled (experimental)
python run_ag_task.py \
    --topics_file data/sample_topics.csv \
    --videos_dir data/videos/ \
    --output_file submissions/team_ag_run2.csv \
    --model_config improved_encoder_decoder \
    --enable_grounding \
    --grounding_config configs/grounding_config.json
```

### Generate Synthetic Training Data
```bash
# Generate synthetic Q&A pairs for training
python generate_synthetic_qa.py \
    --videos_dir data/training_videos/ \
    --output_file data/synthetic_qa_pairs.csv \
    --pairs_per_video 5 \
    --config_file configs/synthetic_qa_config.json
```

## 🏗️ Architecture Overview

### Core Components

```
├── src/ag_task/
│   ├── vqa_model.py          # Abstract model interface + HuggingFace implementations
│   ├── model_configs.py      # Pre-defined model configurations
│   ├── grounding.py          # Grounding and evidence retrieval
│   ├── synthetic_qa.py       # Synthetic data generation
│   └── data_loader.py        # Data loading utilities
├── src/utils/
│   └── video_processing.py   # Video frame extraction
├── configs/                  # Configuration files
├── run_ag_task.py           # Main inference script
└── generate_synthetic_qa.py  # Synthetic data generation script
```

### Model Configurations

| Config Name | Description | Models Used |
|-------------|-------------|-------------|
| `baseline_encoder_decoder` | Basic ViT + T5 model | ViT-Base + FLAN-T5-Base |
| `improved_encoder_decoder` | Larger model for better performance | ViT-Large + FLAN-T5-Large |
| `instructblip` | InstructBLIP multimodal model | InstructBLIP-Vicuna-7B |
| `llava` | LLaVA vision-language model | LLaVA-1.5-7B |
| `grounded_vqa` | VQA with grounding features | ViT + T5 + CLIP grounding |

## 🎯 Research Integration Points

### 1. Multimodal Grounding & Alignment
Based on "[End-to-End Video Question-Answer Generation with Generator-Pretester Network](https://arxiv.org/abs/2101.01447)"

**Implementation Status**: 🚧 Framework ready, models to be integrated

**Key Features**:
- **Retrieval-Augmented Grounding**: Retrieve relevant video frames as evidence for each answer
- **Cross-Verification**: Use pretester approach to validate answers against video content
- **Evidence Tracking**: Frame indices, similarity scores, visual concepts

**Usage**:
```python
# Enable grounding in your pipeline
grounding_config = {
    "enable_retrieval_grounding": True,
    "enable_cross_verification": True,
    "similarity_threshold": 0.7
}
```

### 2. Synthetic Q&A Augmentation
Based on "[LongCaptioning: Unlocking the Power of Long Video Caption Generation](https://arxiv.org/abs/2502.15393)"

**Implementation Status**: ✅ Framework complete, LLM integration ready

**Strategies**:
- **Template-Based**: Predefined question patterns (baseline)
- **LLM-Based**: GPT-4V, Qwen-VL-Chat for diverse Q&A generation
- **Back-Translation**: Convert captions to question-answer pairs

**Usage**:
```python
# Generate synthetic data with multiple strategies
synthetic_config = {
    "enable_template_generation": True,
    "enable_llm_generation": True,  # Requires API key
    "enable_back_translation": True
}
```

### 3. Advanced Model Integration

**Supported Model Families**:
- **Vision-Encoder-Decoder**: ViT + T5/FLAN-T5
- **Unified Multimodal**: LLaVA, InstructBLIP, Qwen-VL
- **Custom Models**: Easy to add via abstract interface

**Adding New Models**:
```python
# Add to src/ag_task/model_configs.py
"your_model": {
    "family": "huggingface",
    "type": "unified_multimodal",
    "model_name": "your-org/your-model",
    "description": "Your model description"
}
```

## 📊 Evaluation & Submission

### TRECVID 2025 Submission Format
```csv
Q_ID, Video_ID, Rank, Answer, Time (sec)
1, tui89Xr_iri, 1, she found a surprise birthday party, 5.2341
1, tui89Xr_iri, 2, she found a party, 5.2341
...
```

### Evaluation Metrics
- **Primary**: NDCG (Normalized Discounted Cumulative Gain)
- **Secondary**: STS, METEOR, BERTScore
- **Efficiency**: Generation time per answer

## 📁 Repository Structure

```
├── src/                      # Source code
│   ├── ag_task/             # Answer Generation implementation
│   ├── mc_task/             # Multiple Choice (TODO)
│   └── utils/               # Shared utilities
├── configs/                 # Configuration files
├── data/                    # Dataset and samples
├── docs/                    # Task documentation
├── notebooks/               # Research notebooks
├── submissions/             # Output submissions
├── run_ag_task.py          # Main AG script
├── generate_synthetic_qa.py # Synthetic data script
└── README.md               # This file
```

## 🔬 Research Roadmap

### Phase 1: Foundation ✅
- [x] Flexible model interface
- [x] Basic VQA pipeline
- [x] Configuration system
- [x] Grounding framework
- [x] Synthetic data framework

### Phase 2: Model Integration 🚧
- [ ] LLaVA integration
- [ ] InstructBLIP integration
- [ ] CLIP-based grounding
- [ ] GPT-4V synthetic generation

### Phase 3: Advanced Features 📋
- [ ] Video-specific models (VideoChatGPT)
- [ ] Temporal reasoning
- [ ] Multi-modal fusion techniques
- [ ] Evaluation framework

## 📚 Key Research Papers

1. **[End-to-End Video Question-Answer Generation with Generator-Pretester Network](https://arxiv.org/abs/2101.01447)** - Generator-Pretester approach for grounding
2. **[LongCaptioning: Unlocking the Power of Long Video Caption Generation](https://arxiv.org/abs/2502.15393)** - Long video understanding and captioning
3. **[Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](https://arxiv.org/abs/2506.11155)** - Advanced evaluation methods

## 🤝 Contributing

This framework is designed for research collaboration. Key extension points:

1. **Add new models**: Implement `BaseVQAModel` interface
2. **Enhance grounding**: Extend `BaseGroundingModule`
3. **Improve synthetic generation**: Add new `BaseSyntheticGenerator`
4. **Evaluation metrics**: Contribute evaluation tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 TRECVID 2025 Specific Notes

- **Test Dataset**: ~2000 YouTube shorts (~30 seconds each)
- **Submission Deadline**: TBA
- **Maximum Submissions**: 3 runs per task per team
- **File Format**: `teamname_ag_run1.csv`, `teamname_ag_run2.csv`, etc.

---

*This framework provides a solid foundation for TRECVID 2025 VQA research while maintaining flexibility for advanced experimentation with grounding, synthetic data, and state-of-the-art multimodal models.*

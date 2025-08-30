# Answer Reranking Pipeline

This script reranks candidate answers using the reranker model from `vqa_model_vllm.py` with video content and optional ASR transcripts.

## Features

- **Smart Reranking**: Uses reranker models from `vqa_model_vllm.py` to generate and rank candidate answers
- **ASR Support**: Optional ASR transcript integration for enhanced reranking context
- **Video-Aware**: Leverages video content for better answer ranking
- **Consistent Structure**: Follows the same pattern as `run_vqa.py` for easy integration

## Usage

### Basic Reranking

```bash
python run_vqa.py \
  --model_config qwen_vl_chat_vllm \
  --videos_dir /brtx/603-nvme1/yweng13/VQA/my_videos \
  --csv_files_dir '/home/dzhang98/code/2025-VQA-AG/testing.dataset.vqa.2025.csv' \
  --num_answers 10 \
  --output submissions/final_results.csv
```

```bash
python run_rerank.py \
    --csv_files_dir ./submissions/final_results.csv \
    --videos_dir /brtx/603-nvme1/yweng13/VQA/my_videos \
    --output_file ./submissions/output_reranked.csv \
    --question_csv ./testing.dataset.vqa.2025.csv
```

## Parameters

### Required Parameters
- `--csv_files_dir`: Path to directory containing CSV files with questions and video IDs
- `--videos_dir`: Directory containing video files
- `--output_file`: Path to save the reranked CSV file

### Optional Parameters
- `--model_config`: Model configuration name (default: "reranker_lora_vllm")
- `--num_frames`: Number of frames to extract from each video (default: 16)
- `--max_videos`: Maximum number of videos to process for testing
- `--asr_json`: Path to ASR transcripts JSON file
- `--rerank_temperature`: Temperature for reranking generation (default: 0.0)
- `--rerank_max_tokens`: Maximum tokens for reranking generation (default: 256)

## Input Format

### CSV Files Format
CSV files should contain these columns:
- `Q_ID`: Question ID
- `Video_ID`: Video ID (corresponds to video filename without extension)
- `Question`: Question text

Example:
```csv
Q_ID,Video_ID,Question
1,video_001,What is the person doing in the video?
2,video_002,Describe the activity shown
```

### ASR JSON Format
ASR file can be in either of these formats:

**Dictionary format**:
```json
{
    "video_001": "transcript text for video 1",
    "video_002": "transcript text for video 2"
}
```

**List format**:
```json
[
    {"Video_ID": "video_001", "transcript": "transcript text for video 1"},
    {"Video_ID": "video_002", "transcript": "transcript text for video 2"}
]
```

## Output Format

The output CSV contains:
- `Q_ID`: Question ID
- `Video_ID`: Video ID
- `Rank`: Answer rank (1 is best)
- `Answer`: Answer text
- `Time (sec)`: Generation time in seconds

## Workflow

1. **Load Topics**: Read questions and video information from CSV files
2. **Initialize Model**: Load reranker model using `create_vqa_model()` from `vqa_model_vllm.py`
3. **Process Videos**: For each video, generate ranked candidate answers using the reranker model
4. **Save Results**: Output reranked results to CSV file

## Model Integration

This script integrates with `vqa_model_vllm.py` by:
- Using `create_vqa_model()` to initialize the reranker model
- Calling `generate_answers()` method to get ranked candidate answers
- Leveraging the model's built-in reranking capabilities

## Example Commands

### Quick Test
```bash
python run_rerank.py \
    --csv_files_dir ./data/csv \
    --videos_dir ./data/videos \
    --output_file test_reranked.csv \
    --max_videos 5
```

### Production Run
```bash
python run_rerank.py \
    --csv_files_dir ./data/csv \
    --videos_dir ./data/videos \
    --output_file production_reranked.csv \
    --asr_json ./data/asr_transcripts.json \
    --rerank_temperature 0.0
```

## Notes

- Ensure sufficient GPU memory for the reranker model
- Video files should be named as `{Video_ID}.mp4` in the videos directory
- The script uses the same data loading pattern as `run_vqa.py`
- ASR transcripts provide additional context for better reranking quality
- The reranker model automatically generates answers in ranked order

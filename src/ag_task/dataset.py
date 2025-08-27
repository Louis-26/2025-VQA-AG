"""
Dataset classes for Answer Generation (AG) model training.
Supports training Qwen2.5-VL to generate multiple answers per question.
"""
import json
import os
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset


class AGTrainingDataset(Dataset):
    """
    Dataset for training Answer Generation model to produce multiple answers.
    
    Each item contains:
    - question: The VQA question
    - asr_transcript: ASR text from the video
    - video_id: Video identifier for loading video frames
    - ground_truth: The correct answer for BERTScore computation
    """
    
    def __init__(self, jsonl_path: str, videos_dir: str) -> None:
        """
        Args:
            jsonl_path: Path to JSONL file with training examples
            videos_dir: Directory containing video files (Video_ID.mp4)
        """
        self.videos_dir = videos_dir
        self.items: List[Dict[str, Any]] = []
        
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.items.append(item)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        
        # Try different video file extensions
        video_id = item['Video_ID']
        possible_paths = [
            os.path.join(self.videos_dir, f"{video_id}.f614.mp4"),  # YouTube-dl format
            os.path.join(self.videos_dir, f"{video_id}.mp4"),       # Standard format
            os.path.join(self.videos_dir, f"{video_id}.mkv"),       # Alternative format
        ]
        
        video_path = None
        for path in possible_paths:
            if os.path.exists(path):
                video_path = path
                break
        
        return {
            "Q_ID": item.get("Q_ID"),
            "Video_ID": item.get("Video_ID"),
            "question": item.get("question", ""),
            "asr_transcript": item.get("asr_transcript", ""),
            "video_path": video_path,
            "ground_truth": item.get("ground_truth", ""),
        }


def build_ag_training_dataset_from_existing(
    json_dir: str,
    output_jsonl: str,
    asr_by_video: Optional[Dict[str, str]] = None
) -> None:
    """
    Build AG training dataset from existing ground truth JSON files.
    
    Args:
        json_dir: Directory containing Video_ID.json files with GT answers
        output_jsonl: Output JSONL file path
        asr_by_video: Optional mapping of Video_ID -> ASR transcript
    """
    asr_by_video = asr_by_video or {}
    
    training_examples = []
    
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue
            
        video_id = filename[:-5]  # Remove .json extension
        json_path = os.path.join(json_dir, filename)
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Extract question and ground truth answer
        question = data.get("question", "")
        ground_truth = data.get("correct_answer", "")
        
        if not question or not ground_truth:
            continue
        
        # Get ASR transcript from external mapping or embedded in JSON
        asr_transcript = (
            asr_by_video.get(video_id, "") or  # External ASR mapping
            data.get("transcript", "") or      # Embedded in JSON
            data.get("asr_transcript", "")     # Alternative field name
        )
        
        # Create training example
        example = {
            "Q_ID": data.get("Q_ID", f"q_{video_id}"),
            "Video_ID": video_id,
            "question": question,
            "asr_transcript": asr_transcript,
            "ground_truth": ground_truth
        }
        
        training_examples.append(example)
    
    # Save to JSONL
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w") as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"Built AG training dataset: {len(training_examples)} examples -> {output_jsonl}")

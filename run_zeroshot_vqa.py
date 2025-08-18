import os
# Mitigate CUDA memory fragmentation unless user already set a policy externally
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import os as _os
import json
from tqdm import tqdm
import pandas as pd

from src.ag_task.json_data_loader import load_json_topics
from src.utils.video_processing import extract_frames
from src.ag_task.vqa_model import create_vqa_model, AnswerCandidate
from src.ag_task.model_configs import get_model_config, list_available_configs

def main():
    parser = argparse.ArgumentParser(description="Run Zero-Shot VQA with a specified model.")
    parser.add_argument("--json_files_dir", type=str, required=True, 
                       help="Path to the directory with JSON files.")
    parser.add_argument("--videos_dir", type=str, required=True, 
                       help="Directory containing the video files.")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save the submission CSV file.")
    parser.add_argument("--model_config", type=str, default="qwen_vl_chat",
                       help="Model configuration name. See model_configs.py for options.")
    parser.add_argument("--num_frames", type=int, default=16, 
                       help="Number of frames to extract from each video.")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process for a quick test.")
    parser.add_argument("--num_answers", type=int, default=16,
                       help="Number of answers to generate per question before de-dup.")
    
    args = parser.parse_args()

    print("=== Zero-Shot VQA Inference Pipeline ===")
    print(f"Model: {args.model_config}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    
    print("\n1. Loading topics from JSON files...")
    topics = load_json_topics(args.json_files_dir)
    if not topics:
        print("No topics found. Exiting.")
        return
        
    if args.max_videos:
        topics = topics[:args.max_videos]
        print(f"Loaded {len(topics)} topics (limited by --max_videos)")
    else:
        print(f"Loaded {len(topics)} topics")

    print("\n2. Initializing model...")
    try:
        model_config = get_model_config(args.model_config)
        vqa_model = create_vqa_model(model_config)
    except ValueError as e:
        print(f"Error: {e}")
        return

    results = []
    print(f"\n3. Processing {len(topics)} videos and generating answers...")
    
    for topic in tqdm(topics, desc="Processing topics"):
        video_id = topic["Video_ID"]
        question = topic["Question"]
        q_id = topic["Q_ID"]
        
        video_path = _os.path.join(args.videos_dir, f"{video_id}.mp4")
        if not _os.path.exists(video_path):
            print(f"Warning: Video file not found for Video_ID {video_id}. Skipping.")
            continue

        try:
            # Pass the video_path directly to the model
            answer_candidates = vqa_model.generate_answers(question=question, video_path=video_path, num_answers=args.num_answers)
        except Exception as e:
            print(f"Warning: Failed to process Video_ID {video_id}: {e}. Skipping.")
            continue
        
        for i, candidate in enumerate(answer_candidates):
            results.append({
                "Q_ID": q_id,
                "Video_ID": video_id,
                "Rank": i + 1,
                "Answer": candidate.text.strip(),
                "Time (sec)": f"{candidate.generation_time:.4f}",
            })

    print(f"\n4. Saving submission file...")
    if results:
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(args.output_file, index=False)
        print(f"Submission file saved to {args.output_file}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()


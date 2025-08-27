#!/usr/bin/env python3
"""
Build AG training dataset from existing ground truth JSON files.
"""
import argparse
import os
from src.ag_task.dataset import build_ag_training_dataset_from_existing


def main():
    parser = argparse.ArgumentParser("Build AG training dataset")
    parser.add_argument("--json_dir", required=True, help="Directory with Video_ID.json GT files")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file path")
    parser.add_argument("--asr_file", help="Optional file with ASR transcripts (JSON format)")
    
    args = parser.parse_args()
    
    # Load ASR data if provided
    asr_by_video = {}
    if args.asr_file and os.path.exists(args.asr_file):
        import json
        with open(args.asr_file, "r") as f:
            asr_data = json.load(f)
            
            # Handle different ASR file formats
            if isinstance(asr_data, list):
                # Format: [{"Video_ID": "vid", "transcript": "text"}, ...]
                for item in asr_data:
                    video_id = item.get("Video_ID")
                    transcript = item.get("transcript", "")
                    if video_id and transcript:
                        asr_by_video[video_id] = transcript
            elif isinstance(asr_data, dict):
                # Format: {"video_id": "transcript", ...}
                asr_by_video = asr_data
            else:
                print(f"Warning: Unsupported ASR file format in {args.asr_file}")
                
        print(f"Loaded ASR data for {len(asr_by_video)} videos")
    
    # Build dataset
    build_ag_training_dataset_from_existing(
        json_dir=args.json_dir,
        output_jsonl=args.output_jsonl,
        asr_by_video=asr_by_video
    )


if __name__ == "__main__":
    main()

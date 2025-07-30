#!/usr/bin/env python3
"""
Synthetic Q&A data generation script for TRECVID 2025 VQA.

This script generates additional training data using various strategies:
1. Template-based generation
2. LLM-based generation (when available)  
3. Back-translation from captions

Based on research from:
- "End-to-End Video Question-Answer Generation with Generator-Pretester Network"
- "LongCaptioning: Unlocking the Power of Long Video Caption Generation"
"""

import argparse
import os
import json
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from src.utils.video_processing import extract_frames
from src.ag_task.synthetic_qa import SyntheticQAPipeline

def load_video_list(videos_dir: str, max_videos: int = None) -> List[Tuple[str, str]]:
    """
    Load list of videos from directory.
    
    Returns:
        List of (video_path, video_id) tuples
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for filename in os.listdir(videos_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in video_extensions:
            video_path = os.path.join(videos_dir, filename)
            video_files.append((video_path, name))
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    return video_files

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A data for video question answering."
    )
    parser.add_argument("--videos_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output CSV file for synthetic Q&A pairs")
    parser.add_argument("--config_file", type=str, default=None,
                       help="JSON configuration file for generation settings")
    parser.add_argument("--pairs_per_video", type=int, default=5,
                       help="Number of Q&A pairs to generate per video")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to extract per video")
    parser.add_argument("--enable_llm", action="store_true",
                       help="Enable LLM-based generation (requires API access)")
    parser.add_argument("--enable_backtrans", action="store_true", 
                       help="Enable back-translation generation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "enable_template_generation": True,  # Always enabled as baseline
        "enable_llm_generation": args.enable_llm,
        "enable_back_translation": args.enable_backtrans,
        "min_confidence": 0.3,
        "template_config": {},
        "llm_config": {
            "model_name": "gpt-4-vision-preview",  # Can be changed
        },
        "backtrans_config": {}
    }
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config.update(json.load(f))
    
    print("=== TRECVID 2025 Synthetic Q&A Generation ===")
    print(f"Videos directory: {args.videos_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Pairs per video: {args.pairs_per_video}")
    print(f"Generation methods enabled:")
    print(f"  - Template-based: True")
    print(f"  - LLM-based: {config['enable_llm_generation']}")
    print(f"  - Back-translation: {config['enable_back_translation']}")
    
    # Initialize synthetic QA pipeline
    print("\n1. Initializing synthetic QA pipeline...")
    pipeline = SyntheticQAPipeline(config)
    
    # Load video files
    print("\n2. Loading video files...")
    video_files = load_video_list(args.videos_dir, args.max_videos)
    print(f"Found {len(video_files)} video files")
    
    if not video_files:
        print("No video files found. Exiting.")
        return
    
    # Process videos and extract frames
    print("\n3. Processing videos and extracting frames...")
    video_data = []
    
    for video_path, video_id in tqdm(video_files, desc="Processing videos"):
        frames = extract_frames(video_path, num_frames=args.num_frames)
        
        if frames.size == 0:
            print(f"Warning: Could not extract frames from {video_path}")
            continue
            
        video_data.append((frames, video_id))
    
    print(f"Successfully processed {len(video_data)} videos")
    
    # Generate synthetic Q&A pairs
    print(f"\n4. Generating synthetic Q&A pairs...")
    synthetic_pairs = pipeline.generate_synthetic_dataset(
        video_data, args.pairs_per_video
    )
    
    print(f"Generated {len(synthetic_pairs)} synthetic Q&A pairs")
    
    # Show generation statistics
    method_counts = {}
    for pair in synthetic_pairs:
        method = pair.generation_method
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("\nGeneration method statistics:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} pairs")
    
    # Save synthetic dataset
    print(f"\n5. Saving synthetic dataset...")
    pipeline.save_synthetic_dataset(synthetic_pairs, args.output_file)
    
    # Save configuration and statistics
    stats_file = args.output_file.replace('.csv', '_stats.json')
    stats = {
        "total_videos_processed": len(video_data),
        "total_pairs_generated": len(synthetic_pairs),
        "pairs_per_video": args.pairs_per_video,
        "generation_methods": method_counts,
        "config": config
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {stats_file}")
    print("\nSynthetic Q&A generation completed successfully!")

if __name__ == "__main__":
    main()
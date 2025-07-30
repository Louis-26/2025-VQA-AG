import argparse
import os
import json
from tqdm import tqdm
import pandas as pd

from src.ag_task.data_loader import load_ag_topics
from src.utils.video_processing import extract_frames
from src.ag_task.vqa_model import create_vqa_model
from src.ag_task.model_configs import get_model_config, list_available_configs
from src.ag_task.grounding import GroundingPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the Answer Generation task for TRECVID 2025 VQA.")
    parser.add_argument("--topics_file", type=str, required=True, 
                       help="Path to the topics CSV file.")
    parser.add_argument("--videos_dir", type=str, required=True, 
                       help="Directory containing the video files.")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save the submission CSV file.")
    parser.add_argument("--model_config", type=str, default="baseline_encoder_decoder",
                       help="Model configuration name. See model_configs.py for options.")
    parser.add_argument("--num_frames", type=int, default=16, 
                       help="Number of frames to extract from each video.")
    parser.add_argument("--num_answers", type=int, default=10, 
                       help="Number of answers to generate per question.")
    parser.add_argument("--enable_grounding", action="store_true",
                       help="Enable grounding and evidence retrieval (experimental).")
    parser.add_argument("--grounding_config", type=str, default=None,
                       help="Path to grounding configuration JSON file.")
    parser.add_argument("--list_configs", action="store_true",
                       help="List available model configurations and exit.")
    
    args = parser.parse_args()

    if args.list_configs:
        print("Available model configurations:")
        configs = list_available_configs()
        for name, desc in configs.items():
            print(f"  {name}: {desc}")
        return

    print("=== TRECVID 2025 VQA Answer Generation Pipeline ===")
    print(f"Model: {args.model_config}")
    print(f"Grounding: {'Enabled' if args.enable_grounding else 'Disabled'}")

    print("\n1. Loading topics...")
    topics = load_ag_topics(args.topics_file)
    if not topics:
        print("No topics found. Exiting.")
        return
    print(f"Loaded {len(topics)} topics")

    print("\n2. Initializing model...")
    try:
        model_config = get_model_config(args.model_config)
        vqa_model = create_vqa_model(model_config)
        print(f"Model initialized: {model_config['description']}")
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list_configs to see available options.")
        return

    print("\n3. Setting up grounding pipeline...")
    grounding_pipeline = None
    if args.enable_grounding:
        grounding_config = {"enable_retrieval_grounding": True}
        if args.grounding_config and os.path.exists(args.grounding_config):
            with open(args.grounding_config, 'r') as f:
                grounding_config.update(json.load(f))
        
        grounding_pipeline = GroundingPipeline(grounding_config)
        print("Grounding pipeline initialized")
    else:
        print("Grounding disabled")
    
    results = []
    print(f"\n4. Processing {len(topics)} videos and generating answers...")
    
    for topic in tqdm(topics, desc="Processing topics"):
        video_id = topic["Video_ID"]
        question = topic["Question"]
        q_id = topic["Q_ID"]
        
        # Support multiple video formats
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_path = None
        
        for ext in video_extensions:
            candidate_path = os.path.join(args.videos_dir, f"{video_id}{ext}")
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        
        if video_path is None:
            print(f"Warning: Video file not found for Video_ID {video_id}. Skipping.")
            continue

        # Extract frames
        frames = extract_frames(video_path, num_frames=args.num_frames)
        if frames.size == 0:
            print(f"Warning: Could not extract frames for Video_ID {video_id}. Skipping.")
            continue
        
        # Generate answer candidates
        answer_candidates = vqa_model.generate_answers(
            frames, question, num_answers=args.num_answers
        )
        
        # Apply grounding if enabled
        if grounding_pipeline is not None:
            answer_texts = [candidate.text for candidate in answer_candidates]
            grounded_pairs = grounding_pipeline.apply_grounding(
                answer_texts, frames, question
            )
            
            # Update candidates with grounding evidence
            for i, (answer_text, evidence) in enumerate(grounded_pairs):
                if i < len(answer_candidates):
                    answer_candidates[i].grounding_evidence = evidence.__dict__
        
        # Format results for submission
        for i, candidate in enumerate(answer_candidates):
            results.append({
                "Q_ID": q_id,
                "Video_ID": video_id,
                "Rank": i + 1,
                "Answer": candidate.text.strip(),
                "Time (sec)": f"{candidate.generation_time:.4f}",
            })

    print(f"\n5. Saving submission file...")
    if results:
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(args.output_file, index=False)
        print(f"Submission file saved to {args.output_file}")
        print(f"Generated {len(results)} total answers for {len(set(r['Q_ID'] for r in results))} questions")
        
        # Save additional metadata if grounding was used
        if args.enable_grounding:
            metadata_file = args.output_file.replace('.csv', '_metadata.json')
            metadata = {
                "model_config": args.model_config,
                "grounding_enabled": True,
                "num_frames": args.num_frames,
                "num_questions": len(topics),
                "total_answers": len(results)
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {metadata_file}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main() 
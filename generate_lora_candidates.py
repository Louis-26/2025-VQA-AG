#!/usr/bin/env python3
"""
Generate answer candidates using LoRA fine-tuned Qwen2.5-VL model.

This script loads the trained LoRA model and generates 10-answer candidates
for the training dataset to evaluate model performance and create reranker data.
"""
import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any

from src.ag_task.model_configs import get_model_config
from src.ag_task.vqa_model_vllm import create_vqa_model
from src.utils.video_processing import extract_frames


def load_training_data(json_dir: str, asr_file: str = None) -> List[Dict[str, Any]]:
    """
    Load training data from JSON files and optional ASR transcripts.
    
    Args:
        json_dir: Directory containing video question JSON files
        asr_file: Optional ASR transcript file
        
    Returns:
        List of training examples with Q_ID, Video_ID, question, ground_truth, asr_transcript
    """
    training_data = []
    asr_data = {}
    
    # Load ASR transcripts if provided
    if asr_file and os.path.exists(asr_file):
        print(f"Loading ASR transcripts from: {asr_file}")
        with open(asr_file, 'r') as f:
            asr_list = json.load(f)
            for item in asr_list:
                asr_data[item['Q_ID']] = item.get('transcript', '')
    
    # Load training questions
    print(f"Loading training data from: {json_dir}")
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in tqdm(json_files, desc="Loading JSON files"):
        video_id = json_file.replace('.json', '')
        json_path = os.path.join(json_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            training_data.append({
                'Q_ID': data['Q_ID'],
                'Video_ID': video_id,
                'question': data['question'],
                'ground_truth': data['correct_answer'],
                'asr_transcript': asr_data.get(data['Q_ID'], '')
            })
            
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Loaded {len(training_data)} training examples")
    return training_data


def generate_candidates_batch(
    model, 
    training_data: List[Dict[str, Any]], 
    videos_dir: str,
    batch_size: int = 1,
    max_examples: int = None
) -> List[Dict[str, Any]]:
    """
    Generate answer candidates for training data using the LoRA model.
    
    Args:
        model: Initialized LoRA VQA model
        training_data: List of training examples
        videos_dir: Directory containing video files
        batch_size: Batch size for processing (currently 1 for video models)
        max_examples: Maximum number of examples to process (None for all)
        
    Returns:
        List of results with Q_ID, Video_ID, Rank, Answer, etc.
    """
    results = []
    
    if max_examples:
        training_data = training_data[:max_examples]
        
    print(f"Generating candidates for {len(training_data)} examples...")
    
    for example in tqdm(training_data, desc="Generating candidates"):
        q_id = example['Q_ID']
        video_id = example['Video_ID']
        question = example['question']
        ground_truth = example['ground_truth']
        asr_transcript = example.get('asr_transcript', '')
        
        # Find video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_path = None
        
        for ext in video_extensions:
            candidate_path = os.path.join(videos_dir, f"{video_id}{ext}")
            if os.path.exists(candidate_path):
                video_path = candidate_path
                break
        
        if video_path is None:
            print(f"Warning: Video file not found for Video_ID {video_id}. Skipping.")
            continue
        
        try:
            # Generate 10 answer candidates
            candidates = model.generate_answers(
                question=question,
                video_path=video_path,
                asr_transcript=asr_transcript,
                num_answers=10
            )
            
            # Format results
            for rank, candidate in enumerate(candidates, 1):
                results.append({
                    'Q_ID': q_id,
                    'Video_ID': video_id,
                    'Rank': rank,
                    'Answer': candidate.text,
                    'Confidence': candidate.confidence,
                    'Generation_Time': candidate.generation_time,
                    'Ground_Truth': ground_truth,
                    'Question': question
                })
                
        except Exception as e:
            print(f"Warning: Failed to generate candidates for {q_id}: {e}")
            continue
    
    return results


def evaluate_candidates(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate generated candidates against ground truth.
    
    Args:
        results: List of generation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        from bert_score import BERTScorer
        
        # Initialize BERTScore
        bertscore = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            device="auto",
            batch_size=64,
            nthreads=4
        )
        
        # Group by Q_ID to evaluate each question
        questions = {}
        for result in results:
            q_id = result['Q_ID']
            if q_id not in questions:
                questions[q_id] = {
                    'answers': [],
                    'ground_truth': result['Ground_Truth']
                }
            questions[q_id]['answers'].append(result['Answer'])
        
        # Calculate metrics
        rank1_bertscore = []
        max_bertscore = []
        
        for q_id, data in tqdm(questions.items(), desc="Evaluating with BERTScore"):
            answers = data['answers']
            ground_truth = data['ground_truth']
            
            if not answers:
                continue
                
            # Compute BERTScore for all answers
            _, _, f1_scores = bertscore.score(answers, [ground_truth] * len(answers))
            f1_scores = f1_scores.tolist()
            
            # Rank 1 performance (first answer)
            rank1_bertscore.append(f1_scores[0])
            
            # Max performance (best of 10 answers)
            max_bertscore.append(max(f1_scores))
        
        metrics = {
            'rank1_bertscore_mean': sum(rank1_bertscore) / len(rank1_bertscore) if rank1_bertscore else 0.0,
            'max_bertscore_mean': sum(max_bertscore) / len(max_bertscore) if max_bertscore else 0.0,
            'num_questions': len(questions),
            'total_candidates': len(results)
        }
        
        return metrics
        
    except ImportError:
        print("Warning: BERTScore not available. Skipping evaluation.")
        return {
            'num_questions': len(set(r['Q_ID'] for r in results)),
            'total_candidates': len(results)
        }


def main():
    parser = argparse.ArgumentParser(description="Generate candidates using LoRA fine-tuned model")
    
    # Data arguments
    parser.add_argument("--json_dir", required=True, help="Directory with training JSON files")
    parser.add_argument("--videos_dir", required=True, help="Directory with training videos")
    parser.add_argument("--asr_file", help="ASR transcript file (optional)")
    parser.add_argument("--output_csv", required=True, help="Output CSV file for candidates")
    
    # Model arguments
    parser.add_argument("--model_config", default="ag_lora_transformers", 
                       choices=["ag_lora_transformers", "ag_lora_vllm"],
                       help="Model configuration to use")
    parser.add_argument("--lora_path", help="Override LoRA adapter path")
    
    # Generation arguments
    parser.add_argument("--max_examples", type=int, help="Maximum examples to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently 1 for video)")
    
    # Evaluation arguments
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip BERTScore evaluation")
    
    args = parser.parse_args()
    
    print("=== LoRA VQA Answer Generation ===")
    print(f"Model: {args.model_config}")
    print(f"JSON dir: {args.json_dir}")
    print(f"Videos dir: {args.videos_dir}")
    print(f"Output: {args.output_csv}")
    
    # Load model configuration
    try:
        model_config = get_model_config(args.model_config)
        
        # Override LoRA path if provided
        if args.lora_path:
            model_config["lora_adapter_path"] = args.lora_path
            
        print(f"Model config: {model_config['description']}")
        print(f"LoRA adapter: {model_config['lora_adapter_path']}")
        
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load training data
    training_data = load_training_data(args.json_dir, args.asr_file)
    if not training_data:
        print("No training data found. Exiting.")
        return
    
    # Initialize model
    print("\nInitializing model...")
    try:
        model = create_vqa_model(model_config)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Generate candidates
    print("\nGenerating answer candidates...")
    results = generate_candidates_batch(
        model=model,
        training_data=training_data,
        videos_dir=args.videos_dir,
        batch_size=args.batch_size,
        max_examples=args.max_examples
    )
    
    if not results:
        print("No candidates generated. Exiting.")
        return
    
    # Save results
    print(f"\nSaving {len(results)} candidates to: {args.output_csv}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    
    # Print summary
    num_questions = len(set(r['Q_ID'] for r in results))
    print(f"\nGeneration Summary:")
    print(f"- Questions processed: {num_questions}")
    print(f"- Total candidates: {len(results)}")
    print(f"- Average candidates per question: {len(results) / num_questions:.1f}")
    
    # Evaluate if requested
    if not args.skip_evaluation:
        print("\nEvaluating candidates...")
        metrics = evaluate_candidates(results)
        
        print(f"\nEvaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"- {metric}: {value:.4f}")
            else:
                print(f"- {metric}: {value}")
        
        # Save evaluation metrics
        metrics_file = args.output_csv.replace('.csv', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_file}")
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()


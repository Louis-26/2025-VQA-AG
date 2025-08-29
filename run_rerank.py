import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import json
from tqdm import tqdm
import pandas as pd

from src.ag_task.json_data_loader import load_json_topics,load_csv_topics
from src.ag_task.vqa_model_vllm import create_vqa_model, AnswerCandidate

from src.ag_task.model_configs import get_model_config, list_available_configs

def load_asr_mapping(asr_json: str) -> dict:
    """Load ASR transcript mapping"""
    mapping = {}
    if not asr_json:
        return mapping
    try:
        with open(asr_json, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            mapping = {str(k): str(v) for k, v in data.items()}
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    vid = item.get("Video_ID") or item.get("video_id")
                    tr = item.get("transcript") or item.get("asr") or item.get("ASR")
                    if vid and tr:
                        mapping[str(vid)] = str(tr)
    except Exception:
        pass
    return mapping

def build_rerank_prompt(question: str, asr: str, candidates: list) -> str:
    """Build reranking prompt"""
    parts = []
    parts.append(f"Question: {question}")
    if asr and asr.strip():
        parts.append(f"ASR Transcript: {asr.strip()}")
    parts.extend([
        "",
        "Please re-rank the following candidate answers from best to worst based on correctness, fidelity to the question, and conciseness.",
        "Output exactly a numbered list 1..N where each line is the exact candidate text.",
        "Do not add commentary.",
        "",
        "Candidates:",
    ])
    for i, a in enumerate(candidates, start=1):
        parts.append(f"{i}. {a}")
    parts.append("")
    parts.append("Ranked list:")
    return "\n".join(parts)

def parse_ranked_list(text: str, candidates: list) -> list:
    """Parse reranking results"""
    try:
        cand_set = set(candidates)
        picked = []
        for line in text.splitlines():

            line = line.strip()
            if not line:
                continue
            # Expect formats like "1. <text>" or just the text
            if line[0].isdigit():
                # Strip leading number and separators
                p = line.split(".", 1)
                if len(p) == 2:
                    line = p[1].strip()
            if line in cand_set and line not in picked:
                picked.append(line)
        if len(picked) != 0:
            return picked, True
        else:
            return picked, False

    except Exception as e:
        print(f"Error parsing ranked list: {e}")
        return [], False

def rerank_with_qwen_vl(model, processor, video_path: str, prompt_text: str, 
                        temperature: float = 0.0, max_new_tokens: int = 512) -> str:
    """Perform reranking using Qwen2.5-VL"""
    # Build messages for Qwen chat
    content = []
    if video_path and os.path.exists(video_path):
        content.append({"type": "video", "video": f"file://{os.path.abspath(video_path)}"})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    # Apply chat template and run generation
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_input], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out_ids = model.generate(
                **inputs,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

    # Trim prompt tokens and decode
    gen_ids_trimmed = out_ids[:, inputs["input_ids"].shape[1]:]
    text_out = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text_out

def main():
    parser = argparse.ArgumentParser(description="Run Answer Reranking with a specified model.")
    parser.add_argument("--csv_files_dir", type=str, required=True, 
                       help="Path to the directory with CSV files.")
    parser.add_argument("--videos_dir", type=str, required=True, 
                       help="Directory containing the video files.")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save the reranked CSV file.")
    parser.add_argument("--model_config", type=str, default="reranker_lora_vllm",
                       help="Model configuration name. See model_configs.py for options.")
    parser.add_argument("--num_frames", type=int, default=16, 
                       help="Number of frames to extract from each video.")
    parser.add_argument("--max_videos", type=int, default=None,
                       help="Maximum number of videos to process for a quick test.")
    parser.add_argument("--asr_json", type=str, default=None,
                       help="Optional transcripts JSON to enrich reranking prompts.")
    parser.add_argument("--question_csv", type=str, default=None,
                       help="Optional question CSV to enrich reranking prompts.")
    

    args = parser.parse_args()

    print("=== Answer Reranking Pipeline ===")
    print(f"Model: {args.model_config}")
    
    print("\n1. Loading topics from CSV files...")
    if args.csv_files_dir.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(args.csv_files_dir, engine="openpyxl")
    else:
        df = pd.read_csv(args.csv_files_dir)
    required_cols = {"Q_ID", "Video_ID", "Answer"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    # Group by (Q_ID,Video_ID)
    groups = df.groupby(["Q_ID", "Video_ID"], sort=False)

    questions = load_csv_topics(args.question_csv)
    questions = {q["Video_ID"]: q["Question"] for q in questions}


    print("\n2. Initializing model...")
    try:
        model_config = get_model_config(args.model_config)
        print(model_config)
        reranker_model = create_vqa_model(model_config)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load ASR mapping if provided
    # asr_by_vid = load_asr_mapping(args.asr_json)
    import whisper
    asr_model = whisper.load_model("turbo")
    results = []
    
    for gi, ((q_id, video_id), g) in tqdm(enumerate(groups)):
        question = questions.get(video_id, "")
        candidates = [str(x).strip() for x in g["Answer"].tolist()]
        if len(candidates) == 0:
            continue
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found for Video_ID {video_id}. Skipping.")
            continue


            # Generate candidate answers using the reranker model
            # asr = asr_by_vid.get(str(video_id), "")
            
        asr = asr_model.transcribe(video_path)['text']
        # Build reranking prompt
        prompt = build_rerank_prompt(question, asr, candidates)
        success = False
        while not success:
            answer_candidates = reranker_model.generate_answers(
                prompt=prompt, 
                video_path=video_path, 
            )

            # Extract candidate answer texts


            ranked,success = parse_ranked_list(answer_candidates[0].outputs[0].text, candidates)



        if len(ranked) < len(candidates):
            remaining = [c for c in candidates if c not in ranked]
            ranked.extend(remaining)

        for rank_idx, ans_text in enumerate(ranked, start=1):
            results.append(
                {
                    "Q_ID": q_id,
                    "Video_ID": video_id,
                    "Rank": rank_idx,
                    "Answer": ans_text,
                }
            )


        
        # Save results with reranked order


    print(f"\n4. Saving reranked submission file...")
    if results:
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(args.output_file, index=False)
        print(f"Reranked submission file saved to {args.output_file}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()


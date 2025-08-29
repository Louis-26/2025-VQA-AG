#!/usr/bin/env python3
"""
Qwen2.5-VL reranker pipeline: given (Q_ID, Video_ID), question, optional ASR, and a
ranked list of candidate answers, ask Qwen2.5‑VL‑7B‑Instruct to re‑rank the list.

Input: a CSV/XLSX with columns: Q_ID,Video_ID,Answer[,Rank]
Question source: JSONs in --json_dir (Video_ID.json with key 'question'), or CSV
ASR source: optional --asr_json (list of dicts or mapping with Video_ID and 'transcript').

Output: CSV with re‑ranked candidates per (Q_ID,Video_ID).

Requires: transformers>=latest per model card guidance.
"""
import argparse
import os
import json
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

import torch
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def load_asr_mapping(asr_json: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
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


def load_questions_by_video(json_dir: str) -> Dict[str, str]:
    q_by_vid: Dict[str, str] = {}
    for name in os.listdir(json_dir):
        if not name.endswith(".json"):
            continue
        vid = name[:-5]
        try:
            with open(os.path.join(json_dir, name), "r") as f:
                data = json.load(f)
            q = data.get("question", "")
            if q:
                q_by_vid[vid] = q
        except Exception:
            continue
    return q_by_vid


def build_prompt(question: str, asr: Optional[str], candidates: List[str]) -> str:
    parts: List[str] = []
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


def parse_ranked_list(text: str, candidates: List[str]) -> List[str]:
    # Map candidates to normalized for exact matching
    cand_set = set(candidates)
    picked: List[str] = []
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
        # Stop if we have all
        if len(picked) == len(candidates):
            break
    return picked


def rerank_single(
    model,
    processor,
    video_path: Optional[str],
    prompt_text: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> str:
    # Build messages for Qwen chat
    content: List[Dict[str, object]] = []
    if video_path and os.path.exists(video_path):
        content.append({"type": "video", "video": f"file://{os.path.abspath(video_path)}"})
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    # Apply chat template and run generation per Qwen model card
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
    ap = argparse.ArgumentParser("Rerank candidates with Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--candidates_csv", required=True, help="CSV/XLSX with Q_ID,Video_ID,Answer[,Rank]")
    ap.add_argument("--json_dir", required=True, help="Directory with Video_ID.json containing question")
    ap.add_argument("--videos_dir", required=True, help="Directory containing <Video_ID>.mp4")
    ap.add_argument("--asr_json", type=str, default=None, help="Optional transcripts JSON to enrich prompts")
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    # Load HF model and processor
    _ = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    # Inputs
    asr_by_vid = load_asr_mapping(args.asr_json)
    q_by_vid = load_questions_by_video(args.json_dir)

    # Read candidate file
    if args.candidates_csv.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(args.candidates_csv, engine="openpyxl")
    else:
        df = pd.read_csv(args.candidates_csv)
    required_cols = {"Q_ID", "Video_ID", "Answer"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Input file must contain columns: {required_cols}")

    # Group by (Q_ID,Video_ID)
    groups = df.groupby(["Q_ID", "Video_ID"], sort=False)
    rows_out: List[Dict[str, object]] = []

    for gi, ((q_id, video_id), g) in enumerate(groups):
        if args.max_examples and gi >= args.max_examples:
            break
        question = q_by_vid.get(str(video_id), "").strip()
        if not question:
            continue
        candidates = [str(x).strip() for x in g["Answer"].tolist()]
        if len(candidates) == 0:
            continue

        # Prefer file:// video if exists
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        asr = asr_by_vid.get(str(video_id), "")

        prompt = build_prompt(question, asr, candidates)

        text_out = rerank_single(
            model,
            processor,
            video_path if os.path.exists(video_path) else None,
            prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        
        ranked = parse_ranked_list(text_out, candidates)

        # If parsing incomplete, append missing in original order
        if len(ranked) < len(candidates):
            remaining = [c for c in candidates if c not in ranked]
            ranked.extend(remaining)

        for rank_idx, ans_text in enumerate(ranked, start=1):
            rows_out.append(
                {
                    "Q_ID": q_id,
                    "Video_ID": video_id,
                    "Rank": rank_idx,
                    "Answer": ans_text,
                }
            )

    if not rows_out:
        print("No outputs produced; check inputs")
        return
    out_df = pd.DataFrame(rows_out)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote reranked CSV: {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()



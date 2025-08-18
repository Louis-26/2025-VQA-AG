#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Dict

import pandas as pd
import numpy as np

try:
    from src.ag_task.critic_reranker import LlavaCriticReranker, CriticImportError
except Exception as e:
    # Allow running from project root without package install
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from src.ag_task.critic_reranker import LlavaCriticReranker, CriticImportError
    except Exception as e2:
        raise

from src.utils.video_processing import extract_frames


def main():
    ap = argparse.ArgumentParser("Rerank Qwen candidates with LLaVA-Critic-7B")
    ap.add_argument("--candidates_csv", required=True, help="CSV with Q_ID,Video_ID,Rank,Answer")
    ap.add_argument("--videos_dir", required=True, help="Directory containing .mp4 files named <Video_ID>.mp4")
    ap.add_argument("--output_csv", required=True, help="Output CSV path for reranked results")
    ap.add_argument("--max_items", type=int, default=None, help="Limit number of Q_ID items to process")
    ap.add_argument("--max_images", type=int, default=8, help="Max sampled frames to feed critic")
    ap.add_argument("--num_frames", type=int, default=64, help="Frames to extract per video before sampling")
    ap.add_argument("--critic_model", default="lmms-lab/llava-critic-7b", help="Critic model HF id")
    args = ap.parse_args()

    # Load candidates and group by Q_ID
    df = pd.read_csv(args.candidates_csv)
    required_cols = {"Q_ID", "Video_ID", "Answer"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"candidates_csv must contain columns: {required_cols}")

    # Keep all rows per Q_ID (multiple answers)
    groups: Dict[str, pd.DataFrame] = dict(tuple(df.groupby("Q_ID", sort=False)))

    # Limit items if requested
    qids = list(groups.keys())
    if args.max_items is not None:
        qids = qids[: args.max_items]

    # Init critic
    try:
        critic = LlavaCriticReranker(
            model_name=args.critic_model,
            max_images=args.max_images,
            conv_template="qwen_1_5",
        )
    except CriticImportError as e:
        print(str(e))
        sys.exit(1)

    out_rows: List[dict] = []

    for qid in qids:
        g = groups[qid]
        video_id = str(g.iloc[0]["Video_ID"])  # assume same video per Q_ID
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Warning: video not found for Q_ID={qid}, Video_ID={video_id}; skipping")
            continue

        # Extract frames
        frames = extract_frames(video_path, num_frames=args.num_frames)
        if frames.size == 0:
            print(f"Warning: could not decode frames for {video_id}; skipping")
            continue

        candidates = [str(a) for a in g["Answer"].tolist()]
        # Score candidates
        scored = critic.score_candidates(frames, question="", candidates=candidates, transcript=None, temperature=0.0)
        # Sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)

        for rank, (ans, score, latency, raw) in enumerate(scored, start=1):
            out_rows.append(
                {
                    "Q_ID": qid,
                    "Video_ID": video_id,
                    "Rank": rank,
                    "Answer": ans,
                    "Critic_Score": score,
                    "Critic_Latency": f"{latency:.4f}",
                }
            )

    if not out_rows:
        print("No items processed; nothing to write.")
        return

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote reranked results to {args.output_csv}")


if __name__ == "__main__":
    main()

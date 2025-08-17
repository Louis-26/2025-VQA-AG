import argparse
import json
import os
from typing import List, Dict

import numpy as np
import pandas as pd

# Prefer PyAV over OpenCV in this env to avoid numpy/cv2 version pin issues
def extract_frames_av(video_path: str, max_decode: int = 256) -> np.ndarray:
    try:
        import av  # type: ignore
    except Exception:
        print("Error: PyAV is not installed in this environment. pip install av")
        return np.array([])

    if not os.path.exists(video_path):
        print(f"Warning: video not found: {video_path}")
        return np.array([])

    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"Warning: failed to open {video_path}: {e}")
        return np.array([])

    frames: List[np.ndarray] = []
    try:
        for frame in container.decode(video=0):
            # Convert to RGB then to BGR to match downstream expectation
            arr_rgb = frame.to_ndarray(format="rgb24")
            arr_bgr = arr_rgb[..., ::-1]
            frames.append(arr_bgr)
            if len(frames) >= max_decode:
                break
    except Exception as e:
        print(f"Warning: decode error on {video_path}: {e}")
    finally:
        try:
            container.close()
        except Exception:
            pass

    if not frames:
        return np.array([])

    return np.asarray(frames)


def load_question(json_dir: str, q_id: str) -> str:
    json_path = os.path.join(json_dir, f"{q_id}.json")
    if not os.path.exists(json_path):
        print(f"Warning: missing JSON for Q_ID {q_id}: {json_path}")
        return ""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        q = data.get("question", "")
        if not isinstance(q, str):
            q = str(q)
        return q
    except Exception as e:
        print(f"Warning: failed to load question for {q_id}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--videos_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--max_decode_frames", type=int, default=256)
    args = parser.parse_args()

    from src.ag_task.critic_reranker import LlavaCriticReranker  # lazy import after arg parsing

    df = pd.read_csv(args.candidates_csv)
    required_cols = {"Q_ID", "Video_ID", "Rank", "Answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in candidates CSV: {sorted(missing)}")

    # For this one-video sample we still code for generality: group by Q_ID, Video_ID
    groups: Dict[tuple, pd.DataFrame] = {k: v.sort_values("Rank") for k, v in df.groupby(["Q_ID", "Video_ID"]) }

    reranker = LlavaCriticReranker(max_images=args.max_images)

    out_rows: List[Dict[str, object]] = []

    for (q_id, video_id), g in groups.items():
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        question = load_question(args.json_dir, str(q_id))
        if not question:
            print(f"Warning: empty question for Q_ID={q_id}; skipping")
            continue

        frames = extract_frames_av(video_path, max_decode=args.max_decode_frames)
        if frames.size == 0:
            print(f"Warning: no frames for {video_id}; skipping")
            continue

        # Collect candidate answers (preserve input rank order)
        candidates = [str(x) for x in g["Answer"].tolist()]

        # Score and rerank
        scored = reranker.score_candidates(
            frames=frames,
            question=question,
            candidates=candidates,
            transcript=None,
            temperature=0.0,
            max_new_tokens=512,
        )
        # Sort by score desc
        scored.sort(key=lambda x: x[1], reverse=True)

        for rank_idx, (ans_text, score, latency, raw) in enumerate(scored, start=1):
            out_rows.append(
                {
                    "Q_ID": q_id,
                    "Video_ID": video_id,
                    "Rank": rank_idx,
                    "Answer": ans_text,
                    "CriticScore": f"{score:.3f}",
                    "CriticLatencySec": f"{latency:.3f}",
                }
            )

    if not out_rows:
        print("No outputs produced; check inputs")
        return

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote reranked CSV: {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()



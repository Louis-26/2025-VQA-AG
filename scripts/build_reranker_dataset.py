import argparse
import json
from typing import Dict
from src.reranker.build_dataset import build_examples, save_jsonl


def main():
    p = argparse.ArgumentParser("Build reranker JSONL from generator CSV and GT JSONs")
    p.add_argument("--candidates_csv", required=True)
    p.add_argument("--json_dir", required=True)
    p.add_argument("--teacher_model", type=str, default=None, help="Sentence-Transformer name for teacher ordering")
    p.add_argument("--asr_json", type=str, default=None, help="Path to ASR transcripts JSON (list of dicts or mapping)")
    p.add_argument("--out_jsonl", required=True)
    args = p.parse_args()

    # Load ASR mapping if provided
    asr_by_video: Dict[str, str] = {}
    if args.asr_json:
        try:
            with open(args.asr_json, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Expect mapping video_id -> transcript
                asr_by_video = {str(k): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                # Expect list of dicts, each maybe with Video_ID and transcript
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    vid = item.get("Video_ID") or item.get("video_id")
                    tr = item.get("transcript") or item.get("asr") or item.get("ASR")
                    if vid and tr:
                        asr_by_video[str(vid)] = str(tr)
            else:
                print("Warning: Unrecognized ASR JSON structure; skipping ASR")
        except Exception as e:
            print(f"Warning: Failed to load ASR JSON: {e}")
            asr_by_video = {}

    examples = build_examples(
        csv_candidates=args.candidates_csv,
        json_dir=args.json_dir,
        asr_by_video=asr_by_video,
        teacher_model=args.teacher_model,
    )
    save_jsonl(examples, args.out_jsonl)
    print(f"Wrote {len(examples)} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()



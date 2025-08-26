import argparse
from src.reranker.build_dataset import build_examples, save_jsonl


def main():
    p = argparse.ArgumentParser("Build reranker JSONL from generator CSV and GT JSONs")
    p.add_argument("--candidates_csv", required=True)
    p.add_argument("--json_dir", required=True)
    p.add_argument("--teacher_model", type=str, default=None, help="Sentence-Transformer name for teacher ordering")
    p.add_argument("--out_jsonl", required=True)
    args = p.parse_args()

    examples = build_examples(
        csv_candidates=args.candidates_csv,
        json_dir=args.json_dir,
        asr_by_video=None,
        teacher_model=args.teacher_model,
    )
    save_jsonl(examples, args.out_jsonl)
    print(f"Wrote {len(examples)} examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()



import argparse
import json
import random
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser("Split a JSONL into train/dev")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--out_train_jsonl", required=True)
    p.add_argument("--out_dev_jsonl", required=True)
    p.add_argument("--dev_size", type=int, default=100, help="Number of examples in dev set")
    p.add_argument("--dev_ratio", type=float, default=None, help="Alternatively, ratio for dev split (e.g., 0.2)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    items = load_jsonl(args.input_jsonl)
    n = len(items)
    if args.dev_ratio is not None:
        dev_size = max(1, int(round(n * args.dev_ratio)))
    else:
        dev_size = min(args.dev_size, n - 1 if n > 1 else 1)

    random.Random(args.seed).shuffle(items)
    dev = items[:dev_size]
    train = items[dev_size:]

    save_jsonl(train, args.out_train_jsonl)
    save_jsonl(dev, args.out_dev_jsonl)
    print({
        "total": n,
        "train": len(train),
        "dev": len(dev),
        "seed": args.seed,
    })


if __name__ == "__main__":
    main()



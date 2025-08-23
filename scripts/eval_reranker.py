import argparse
import json
import math
from tqdm import tqdm
from typing import List, Dict

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from src.reranker.tokens import add_special_tokens_to_tokenizer
from src.reranker.decoding import greedy_pointer_decode
from src.reranker.constants import R_MAX_RANKS


def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def ndcg_at_k(pred: List[str], gold: str, k: int = 10) -> float:
    # Relevance 1 for gold match, 0 otherwise
    dcg = 0.0
    for i, cid in enumerate(pred[:k], start=1):
        rel = 1.0 if cid == gold else 0.0
        if rel:
            dcg += 1.0 / (math.log2(i + 1))
    idcg = 1.0  # best case: gold at rank 1 => 1/log2(2)=1
    return dcg / idcg


def main():
    p = argparse.ArgumentParser("Evaluate reranker on JSONL using constrained decode")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapters", required=False, help="LoRA adapters path (optional)")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--max_examples", type=int, default=200)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    add_special_tokens_to_tokenizer(tok, None)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True
    )
    if args.adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapters)
    model.eval()

    data = load_jsonl(args.jsonl)
    if args.max_examples:
        data = data[:args.max_examples]
    import math
    n = 0
    top1 = 0
    ndcg_sum = 0.0
    valid_lists = 0
    for ex in tqdm(data, desc="Evaluating"):
        prompt = ex["prompt"]
        # gold candidate id: find which <CAND_*> maps to gold text in the prompt lines
        gold_text = ex["gold"].strip()
        gold_cid = None
        for line in prompt.splitlines():
            if line.startswith("<CAND_"):
                parts = line.split(":", 1)
                if len(parts) == 2 and parts[1].strip() == gold_text:
                    gold_cid = parts[0]
                    break
        if gold_cid is None:
            continue

        pred_cids = greedy_pointer_decode(model, tok, prompt, R_MAX_RANKS)
        # validity: well-formed (no repeats)
        if len(pred_cids) == len(set(pred_cids)) and len(pred_cids) > 0:
            valid_lists += 1
        # metrics
        n += 1
        if pred_cids and pred_cids[0] == gold_cid:
            top1 += 1
        ndcg_sum += ndcg_at_k(pred_cids, gold_cid, k=10)

    print({
        "num_examples": n,
        "top1": top1 / max(1, n),
        "ndcg@10": ndcg_sum / max(1, n),
        "validity": valid_lists / max(1, n),
    })


if __name__ == "__main__":
    main()



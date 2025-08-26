import argparse
import json
import math
from tqdm import tqdm
from typing import List, Dict
import torch
import random

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from src.reranker.tokens import add_special_tokens_to_tokenizer
from src.reranker.decoding import greedy_pointer_decode, extract_present_candidate_tokens_from_prompt
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


def _batched_pointer_decode(model, tokenizer, prompts: List[str], max_ranks: int, batch_size: int) -> List[List[str]]:
    device = next(model.parameters()).device
    # Precompute candidate sets and initialize state per example
    present_tok_strs = [extract_present_candidate_tokens_from_prompt(p) for p in prompts]
    present_ids = [list(map(tokenizer.convert_tokens_to_ids, toks)) for toks in present_tok_strs]
    chosen_ids = [set() for _ in prompts]
    scaffolds = [p for p in prompts]
    emitted: List[List[str]] = [[] for _ in prompts]

    for k in range(1, max_ranks + 1):
        rank_tok = f"<R{k}>"
        active_indices = [i for i in range(len(prompts)) if any(tid not in chosen_ids[i] for tid in present_ids[i])]
        if not active_indices:
            break
        for b_start in range(0, len(active_indices), batch_size):
            batch_idx = active_indices[b_start:b_start + batch_size]
            batch_inputs = [f"{scaffolds[i]}\n{rank_tok}=" for i in batch_idx]
            enc = tokenizer(batch_inputs, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    out = model(**enc)
                    logits = out.logits  # (B, T, V)
                    last_logits = logits[:, -1, :]  # (B, V)
            for bi, i in enumerate(batch_idx):
                remaining = [tid for tid in present_ids[i] if tid not in chosen_ids[i]]
                if not remaining:
                    continue
                rem = torch.tensor(remaining, device=last_logits.device, dtype=torch.long)
                scores = last_logits[bi, rem]
                best_idx = torch.argmax(scores).item()
                next_tid = remaining[best_idx]
                chosen_ids[i].add(next_tid)
                tok_str = tokenizer.convert_ids_to_tokens(next_tid)
                emitted[i].append(tok_str)
                scaffolds[i] = f"{batch_inputs[bi]}{tok_str}\n"
    return emitted


def main():
    p = argparse.ArgumentParser("Evaluate reranker on JSONL using constrained decode")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapters", required=False, help="LoRA adapters path (optional)")
    p.add_argument("--jsonl", required=True)
    p.add_argument("--max_examples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
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
    # Prepare prompts and gold ids upfront
    prompts = [ex["prompt"] for ex in data]
    gold_ids = []
    for ex in data:
        prompt = ex["prompt"]
        gold_text = ex["gold"].strip()
        gold_cid = None
        for line in prompt.splitlines():
            if line.startswith("<CAND_"):
                parts = line.split(":", 1)
                if len(parts) == 2 and parts[1].strip() == gold_text:
                    gold_cid = parts[0]
                    break
        gold_ids.append(gold_cid)

    # Shuffle candidate display order deterministically per example to avoid bias in tie-breaking
    rng = random.Random(42)
    shuffled_prompts = []
    for p in prompts:
        lines = p.splitlines()
        head = []
        cand = []
        tail = []
        mode = "head"
        for line in lines:
            if line.strip().startswith("Candidates:"):
                head.append(line)
                mode = "cand"
                continue
            if mode == "cand":
                if line.strip().startswith("Reasoning:"):
                    tail.append(line)
                    mode = "tail"
                elif line.strip().startswith("<CAND_"):
                    cand.append(line)
                else:
                    tail.append(line)
                    mode = "tail"
            elif mode == "head":
                head.append(line)
            else:
                tail.append(line)
        rng.shuffle(cand)
        shuffled_prompts.append("\n".join(head + cand + tail))

    preds = _batched_pointer_decode(model, tok, shuffled_prompts, R_MAX_RANKS, args.batch_size)
    for pred_cids, gold_cid in tqdm(zip(preds, gold_ids), total=len(gold_ids), desc="Scoring"):
        if gold_cid is None:
            continue
        if len(pred_cids) == len(set(pred_cids)) and len(pred_cids) > 0:
            valid_lists += 1
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



#!/usr/bin/env python3
import argparse
import os
import json
import re
from typing import List, Dict, Any, Tuple

import pandas as pd

# Optional-heavy libs are imported lazily with clear error messages

def safe_imports():
    errs = []
    modules = {}
    try:
        from rouge_score import rouge_scorer
        modules["rouge_scorer"] = rouge_scorer
    except Exception as e:
        errs.append("rouge-score (pip install rouge-score)")
    try:
        import nltk
        from nltk.translate.meteor_score import single_meteor_score
        modules["nltk"] = nltk
        modules["single_meteor_score"] = single_meteor_score
    except Exception:
        errs.append("nltk (pip install nltk)")
    try:
        from bert_score import score as bert_score
        modules["bert_score"] = bert_score
    except Exception:
        errs.append("bert-score (pip install bert-score)")
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        modules["SentenceTransformer"] = SentenceTransformer
        modules["np"] = np
    except Exception:
        errs.append("sentence-transformers (pip install sentence-transformers)")
    
    try:
        from sklearn.metrics import ndcg_score
        modules["ndcg_score"] = ndcg_score
    except Exception:
        errs.append("scikit-learn (pip install scikit-learn)")

    if errs:
        raise RuntimeError(
            "Missing required packages: " + ", ".join(errs)
        )
    return modules



def ensure_nltk_data(nltk_mod):
    try:
        nltk_mod.data.find("corpora/wordnet")
    except LookupError:
        nltk_mod.download("wordnet", quiet=True)
    try:
        nltk_mod.data.find("corpora/omw-1.4")
    except LookupError:
        nltk_mod.download("omw-1.4", quiet=True)


def load_ground_truths(json_dir: str) -> Dict[str, str]:
    """Map Q_ID (filename stem) -> correct_answer string."""
    mapping: Dict[str, str] = {}
    for fname in os.listdir(json_dir):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        fpath = os.path.join(json_dir, fname)
        try:
            with open(fpath, "r") as f:
                obj = json.load(f)
            gt = obj.get("correct_answer")
            if isinstance(gt, str) and gt.strip():
                mapping[stem] = gt.strip()
        except Exception:
            continue
    return mapping


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation (portable), collapse spaces."""
    s = s.lower()
    # Remove any non-word, non-space characters
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_rouge_l(preds: List[str], refs: List[str], rouge_scorer):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    per_item = []
    for p, r in zip(preds, refs):
        sc = scorer.score(r, p)["rougeL"].fmeasure
        per_item.append(sc)
    return per_item, sum(per_item) / len(per_item) if per_item else 0.0


def compute_meteor(preds: List[str], refs: List[str], single_meteor_score):
    def _tok(s: str) -> List[str]:
        # Simple whitespace tokenization to satisfy NLTK METEOR's Iterable[str] requirement
        return s.strip().split()
    scores = [single_meteor_score(_tok(r), _tok(p)) for p, r in zip(preds, refs)]
    return scores, sum(scores) / len(scores) if scores else 0.0


def compute_bertscore(
    preds: List[str],
    refs: List[str],
    bert_score,
    model_type: str = None,
    rescale_with_baseline: bool = True,
    use_idf: bool = False,
):
    kwargs: Dict[str, Any] = {"lang": "en", "rescale_with_baseline": rescale_with_baseline}
    if model_type:
        kwargs["model_type"] = model_type
        kwargs.pop("lang", None)  # model_type overrides lang
    if use_idf:
        kwargs["idf"] = True
    P, R, F1 = bert_score(preds, refs, **kwargs)
    f1_list = F1.tolist()
    return f1_list, sum(f1_list) / len(f1_list) if f1_list else 0.0


def compute_sts(preds: List[str], refs: List[str], SentenceTransformer, np, model_name: str):
    model = SentenceTransformer(model_name)
    emb_p = model.encode(preds, convert_to_numpy=True, normalize_embeddings=True)
    emb_r = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    sims = (emb_p * emb_r).sum(axis=1)  # cosine since normalized
    sims_list = sims.tolist()
    return sims_list, float(np.mean(sims)) if len(sims_list) else 0.0


def main():
    parser = argparse.ArgumentParser("Evaluate generated AG answers against JSON ground truth")
    parser.add_argument("--pred_file", required=True, help="CSV with columns Q_ID,Answer (others ignored)")
    parser.add_argument("--json_files_dir", required=True, help="Directory of ground-truth JSON files")
    parser.add_argument("--per_item_out", default=None, help="Optional path to write per-item metrics CSV")
    parser.add_argument("--summary_out", default=None, help="Optional path to write summary JSON")

    # Sanity-check options
    parser.add_argument("--normalize", action="store_true", help="Also compute metrics on normalized text")
    parser.add_argument("--bertscore_model", default=None, help="Override model_type for BERTScore (e.g., roberta-large)")
    parser.add_argument("--no_bertscore_rescale", action="store_true", help="Disable rescale_with_baseline for BERTScore")
    parser.add_argument("--bertscore_use_idf", action="store_true", help="Enable IDF weighting for BERTScore")
    parser.add_argument("--sts_model", default="sentence-transformers/all-MiniLM-L6-v2", help="STS model (e.g., sentence-transformers/all-mpnet-base-v2)")

    args = parser.parse_args()

    mods = safe_imports()
    nltk_mod = mods.get("nltk")
    ensure_nltk_data(nltk_mod)

    # Load predictions
    df = pd.read_csv(args.pred_file)
    if "Q_ID" not in df.columns or "Answer" not in df.columns:
        raise ValueError("pred_file must contain columns Q_ID and Answer")

    # Keep first prediction per Q_ID if duplicates
    # df = df.sort_values(by=["Q_ID", "CriticScore","CriticLatencySec"], ascending=[True, False, True]) if "Rank" in df.columns else df
    df_unique = df.drop_duplicates(subset=["Q_ID"], keep="first").reset_index(drop=True)

    # Load ground truth
    gt_map = load_ground_truths(args.json_files_dir)
    # Align
    preds, refs, qids = [], [], []
    missing = 0
    for _, row in df_unique.iterrows():
        qid = str(row["Q_ID"]).strip()
        pred = str(row["Answer"]).strip()

        ref = gt_map.get(row["Video_ID"])
        if ref is None:
            missing += 1
            continue
        qids.append(qid)
        preds.append(pred)
        refs.append(ref)

    if not preds:
        raise RuntimeError("No aligned predictions with ground-truth were found. Check Q_ID matching.")

    # Compute metrics (raw)
    rouge_list, rouge_avg = compute_rouge_l(preds, refs, mods["rouge_scorer"]) 
    meteor_list, meteor_avg = compute_meteor(preds, refs, mods["single_meteor_score"]) 
    bert_list, bert_avg = compute_bertscore(
        preds, refs, mods["bert_score"],
        model_type=args.bertscore_model,
        rescale_with_baseline=not args.no_bertscore_rescale,
        use_idf=args.bertscore_use_idf,
    )
    sts_list, sts_avg = compute_sts(preds, refs, mods["SentenceTransformer"], mods["np"], args.sts_model) 

    per_item_dict = {
        "Q_ID": qids,
        "Prediction": preds,
        "Reference": refs,
        "ROUGE_L": rouge_list,
        "METEOR": meteor_list,
        "BERTScore_F1": bert_list,
        "STS_Cosine": sts_list,
    }

    summary = {
        "num_items_evaluated": len(qids),
        "num_missing_ground_truth": int(missing),
        "averages": {
            "ROUGE_L": rouge_avg,
            "METEOR": meteor_avg,
            "BERTScore_F1": bert_avg,
            "STS_Cosine": sts_avg,
        }
    }

    # Optionally compute normalized metrics
    if args.normalize:
        preds_n = [normalize_text(x) for x in preds]
        refs_n = [normalize_text(x) for x in refs]
        rouge_n_list, rouge_n_avg = compute_rouge_l(preds_n, refs_n, mods["rouge_scorer"]) 
        meteor_n_list, meteor_n_avg = compute_meteor(preds_n, refs_n, mods["single_meteor_score"]) 
        bert_n_list, bert_n_avg = compute_bertscore(
            preds_n, refs_n, mods["bert_score"],
            model_type=args.bertscore_model,
            rescale_with_baseline=not args.no_bertscore_rescale,
            use_idf=args.bertscore_use_idf,
        )
        sts_n_list, sts_n_avg = compute_sts(preds_n, refs_n, mods["SentenceTransformer"], mods["np"], args.sts_model) 

        per_item_dict.update({
            "ROUGE_L_norm": rouge_n_list,
            "METEOR_norm": meteor_n_list,
            "BERTScore_F1_norm": bert_n_list,
            "STS_Cosine_norm": sts_n_list,
        })
        summary["averages_norm"] = {
            "ROUGE_L": rouge_n_avg,
            "METEOR": meteor_n_avg,
            "BERTScore_F1": bert_n_avg,
            "STS_Cosine": sts_n_avg,
        }

    # Outputs
    if args.per_item_out is None:
        base = os.path.splitext(args.pred_file)[0]
        args.per_item_out = base + "_per_item_metrics.csv"
    if args.summary_out is None:
        base = os.path.splitext(args.pred_file)[0]
        args.summary_out = base + "_summary.json"

    per_item = pd.DataFrame(per_item_dict)
    per_item.to_csv(args.per_item_out, index=False)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Evaluation Complete ===")
    print(f"Evaluated items: {summary['num_items_evaluated']}")
    print(f"Missing ground-truth: {summary['num_missing_ground_truth']}")
    print("Averages (raw):")
    for k, v in summary["averages"].items():
        print(f"  {k}: {v:.4f}")
    if args.normalize:
        print("Averages (normalized):")
        for k, v in summary["averages_norm"].items():
            print(f"  {k}: {v:.4f}")
    print(f"Per-item metrics: {args.per_item_out}")
    print(f"Summary JSON: {args.summary_out}")


if __name__ == "__main__":
    main()

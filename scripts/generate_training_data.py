import argparse
import json
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
def compute_sts(preds: List[str], refs: List[str], SentenceTransformer, model_name: str):
    model = SentenceTransformer(model_name)
    emb_p = model.encode(preds, convert_to_numpy=True, normalize_embeddings=True)
    emb_r = model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    sims = (emb_p * emb_r).sum(axis=1)  # cosine since normalized

    return sims, float(np.mean(sims))  


def insert_gt_and_drop_min(ans:List[str], scores:np.ndarray, ground_truth:str, gt_score:float=None, seed:int=None) -> Tuple[List[str], np.ndarray]:
    """
    ans: list, length N (e.g. 10)
    scores: np.ndarray of shape (N,)
    ground_truth: the element to insert into ans
    gt_score: optional, the score associated with ground_truth (default = np.nan)
    seed: optional, random seed for reproducibility
    
    Returns:
        new_ans: list, length still N
        new_scores: np.ndarray, shape (N,)
    """
    ans = list(ans)  # copy to avoid modifying in place
    scores = np.asarray(scores)

    # Step 1: remove the answer with the lowest score
    # drop_idx = int(scores.argmin())
    # ans.pop(drop_idx)
    # scores = np.delete(scores, drop_idx)

    # Step 2: insert ground truth at a random position
    rng = np.random.default_rng(seed)
    insert_pos = int(rng.integers(0, len(ans) + 1))
    ans.insert(insert_pos, ground_truth)

    # Step 3: insert corresponding score (default np.nan if not provided)
    if gt_score is None:
        gt_score = np.nan
    scores = np.insert(scores, insert_pos, gt_score)

    return ans, scores

def rank_candidates(
    candidates: List[str],
    ground_truth: str,
    fake_answers: List[str],
    model_cls,
    model_name: str = "all-MiniLM-L6-v2",
    weight_gt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Return the rank list, the final score list, and the score of the ground truth
    """
    n = len(candidates)
    gt_refs = [ground_truth] * n
    score_gt, _ = compute_sts(candidates, gt_refs, model_cls, model_name)

    fake_mat = np.array([
        compute_sts(candidates, [ans] * n, model_cls, model_name)[0]
        for ans in fake_answers
    ])  # shape: (k, n)

    final = weight_gt * score_gt - fake_mat.mean(axis=0)

    candidates, final = insert_gt_and_drop_min(candidates, final, ground_truth, weight_gt)

    order = np.argsort(final)[::-1]
    return order, final, candidates

def load_question(json_dir: str, q_id: str) -> str:
    json_path = os.path.join(json_dir, f"{q_id}.json")
    if not os.path.exists(json_path):
        print(f"Warning: missing JSON for Q_ID {q_id}: {json_path}")
        return ""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        q = data
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
    args = parser.parse_args()

    df = pd.read_csv(args.candidates_csv)
    required_cols = {"Q_ID", "Video_ID", "Rank", "Answer"}
    missing = required_cols - set(df.columns)
    if missing: 
        raise ValueError(f"Missing required columns in candidates CSV: {sorted(missing)}")

    # For this one-video sample we still code for generality: group by Q_ID, Video_ID
    groups: Dict[tuple, pd.DataFrame] = {k: v.sort_values("Rank") for k, v in df.groupby(["Q_ID", "Video_ID"]) }


    out_rows: List[Dict[str, object]] = []

    for (q_id, video_id), g in tqdm(groups.items()):
        video_path = os.path.join(args.videos_dir, f"{video_id}.mp4")
        data = load_question(args.json_dir, str(q_id))
        ground_truth = data["correct_answer"]
        fake_answer = data["incorrect_answers"]
        candidates = [str(x) for x in g["Answer"].tolist()]

        order, final, candidates = rank_candidates(candidates, ground_truth, fake_answer, SentenceTransformer, "all-MiniLM-L6-v2", 1.0)
        if not data:
            print(f"Warning: empty question for Q_ID={q_id}; skipping")
            continue

        out_rows.append(
            {
                "Q_ID": q_id,
                "Video_ID": video_id,
                "Answer List":candidates,
                "Rank List": order,
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



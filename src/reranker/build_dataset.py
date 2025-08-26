import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import math

import pandas as pd

from .prompter import enumerate_and_shuffle_candidates, build_pointer_list_target, format_reranker_prompt
from .constants import R_MAX_RANKS


@dataclass
class RerankerExample:
    q_id: str
    video_id: str
    question: str
    asr: str
    candidates: List[str]
    gold_answer: str
    wrong_answers: List[str]
    # Targets
    target_rank_ids: List[str]
    rank_text: str
    prompt_text: str


def _load_gt(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_jsons_by_video(json_dir: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for name in os.listdir(json_dir):
        if not name.endswith(".json"):
            continue
        vid = name[:-5]
        mapping[vid] = _load_gt(os.path.join(json_dir, name))
    return mapping


def _collect_generated_answers(csv_path: str) -> Dict[Tuple[str, str], List[str]]:
    """Load candidates from CSV or Excel (.xlsx/.xls).

    - For CSV: robust parsing with python engine and skipping bad lines.
    - For Excel: read first sheet and select required columns.
    """
    lower = csv_path.lower()
    usecols = ["Q_ID", "Video_ID", "Rank", "Answer"]
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        try:
            df = pd.read_excel(csv_path, engine="openpyxl", usecols=usecols, dtype={
                "Q_ID": str, "Video_ID": str, "Rank": int, "Answer": str,
            })
        except ImportError as e:
            raise RuntimeError("Reading .xlsx requires openpyxl. Please install: pip install openpyxl") from e
    else:
        # CSV path
        df = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            usecols=usecols,
            dtype={"Q_ID": str, "Video_ID": str, "Rank": int, "Answer": str},
        )
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for _, row in df.iterrows():
        key = (row["Q_ID"], row["Video_ID"])
        grouped.setdefault(key, []).append(str(row["Answer"]).strip())
    return grouped


def _rank_tail_by_teacher_or_random(
    cand_ids: List[str],
    gold_cand_id: str,
    id_to_text: Dict[str, str],
    teacher: Optional["TeacherScorer"] = None,
) -> List[str]:
    import random
    others = [c for c in cand_ids if c != gold_cand_id]
    if teacher is None:
        random.shuffle(others)
        return others
    # Score by similarity to gold answer text; higher is better
    gold_text = id_to_text.get(gold_cand_id, "").strip()
    scores = teacher.score_batch([id_to_text[o] for o in others], gold_text)
    ranked = [x for _, x in sorted(zip(scores, others), key=lambda t: t[0], reverse=True)]
    return ranked


class TeacherScorer:
    """Light wrapper for sentence-embedding similarity as teacher signal."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None

    def _ensure(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise RuntimeError("Teacher scoring requires sentence-transformers. pip install sentence-transformers") from e
            self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def _cos_sim(a, b):
        import numpy as np
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float((a * b).sum())

    def score_batch(self, candidates: List[str], gold_text: str) -> List[float]:
        self._ensure()
        embs = self.model.encode([gold_text] + candidates, normalize_embeddings=False, show_progress_bar=False)
        gold_emb = embs[0]
        cand_embs = embs[1:]
        return [self._cos_sim(c, gold_emb) for c in cand_embs]


def build_examples(
    csv_candidates: str,
    json_dir: str,
    asr_by_video: Optional[Dict[str, str]] = None,
    teacher_model: Optional[str] = None,
) -> List[RerankerExample]:
    asr_by_video = asr_by_video or {}
    gen_by_key = _collect_generated_answers(csv_candidates)
    meta_by_video = _load_jsons_by_video(json_dir)

    examples: List[RerankerExample] = []
    teacher = TeacherScorer(teacher_model) if teacher_model else None
    for (q_id, video_id), gen_answers in gen_by_key.items():
        meta = meta_by_video.get(video_id)
        if not meta:
            continue
        question = meta.get("question", "")
        gold = meta.get("correct_answer", "").strip()
        wrongs = [w.strip() for w in meta.get("incorrect_answers", [])]

        # Candidate pool: gold + generator candidates (exclude provided human wrongs)
        pool = [gold] + gen_answers
        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for t in pool:
            if t in seen:
                continue
            seen.add(t)
            deduped.append(t)

        cand_lines, id_to_text = enumerate_and_shuffle_candidates(deduped)
        # Find cand_id corresponding to gold
        gold_cid = None
        for cid, text in id_to_text.items():
            if text == gold:
                gold_cid = cid
                break
        if gold_cid is None:
            # If gold text got lost by dedup (shouldn't), skip
            continue
        ranked_tail = _rank_tail_by_teacher_or_random(list(id_to_text.keys()), gold_cid, id_to_text, teacher)
        pointer_lines = build_pointer_list_target(gold_cid, ranked_tail, max_ranks=R_MAX_RANKS)
        prompt = format_reranker_prompt(
            question=question,
            asr_transcript=asr_by_video.get(video_id, ""),
            candidate_texts=deduped,
            candidate_lines=cand_lines,
        )

        examples.append(
            RerankerExample(
                q_id=q_id,
                video_id=video_id,
                question=question,
                asr=asr_by_video.get(video_id, ""),
                candidates=deduped,
                gold_answer=gold,
                wrong_answers=wrongs,
                target_rank_ids=pointer_lines[:-1],  # exclude <ENDLIST>
                rank_text="\n".join(pointer_lines),
                prompt_text=prompt,
            )
        )
    return examples


def save_jsonl(examples: List[RerankerExample], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            obj = {
                "Q_ID": ex.q_id,
                "Video_ID": ex.video_id,
                "question": ex.question,
                "asr": ex.asr,
                "candidates": ex.candidates,
                "gold": ex.gold_answer,
                "wrong": ex.wrong_answers,
                "prompt": ex.prompt_text,
                "target": ex.rank_text,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")



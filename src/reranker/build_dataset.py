import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

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
    df = pd.read_csv(csv_path)
    grouped: Dict[Tuple[str, str], List[str]] = {}
    for _, row in df.iterrows():
        key = (row["Q_ID"], row["Video_ID"])
        grouped.setdefault(key, []).append(str(row["Answer"]).strip())
    return grouped


def _rank_tail_by_teacher_or_random(
    cand_ids: List[str],
    gold_cand_id: str,
) -> List[str]:
    # Placeholder: random tail after GT. Can be replaced with teacher scores.
    import random
    others = [c for c in cand_ids if c != gold_cand_id]
    random.shuffle(others)
    return others


def build_examples(
    csv_candidates: str,
    json_dir: str,
    asr_by_video: Optional[Dict[str, str]] = None,
) -> List[RerankerExample]:
    asr_by_video = asr_by_video or {}
    gen_by_key = _collect_generated_answers(csv_candidates)
    meta_by_video = _load_jsons_by_video(json_dir)

    examples: List[RerankerExample] = []
    for (q_id, video_id), gen_answers in gen_by_key.items():
        meta = meta_by_video.get(video_id)
        if not meta:
            continue
        question = meta.get("question", "")
        gold = meta.get("correct_answer", "").strip()
        wrongs = [w.strip() for w in meta.get("incorrect_answers", [])]

        # Candidate pool: gold + human wrongs + generator K (dedup, keep text)
        pool = [gold] + wrongs + gen_answers
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
        ranked_tail = _rank_tail_by_teacher_or_random(list(id_to_text.keys()), gold_cid)
        pointer_lines = build_pointer_list_target(gold_cid, ranked_tail, max_ranks=R_MAX_RANKS)
        prompt = format_reranker_prompt(
            question=question,
            asr_transcript=asr_by_video.get(video_id, ""),
            candidate_texts=deduped,
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



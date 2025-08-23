import random
from typing import Dict, List, Tuple, Optional

from .constants import (
    R_MAX_RANKS,
    REASONING_PREFIX,
    FINAL_LIST_HEADER,
    RANK_TOKEN_TEMPLATE,
    CAND_TOKEN_TEMPLATE,
    ENDLIST_TOKEN,
)


def enumerate_and_shuffle_candidates(raw_candidates: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Assign randomized candidate IDs <CAND_i> to input candidate texts.

    Returns (lines, mapping) where lines are like "<CAND_3>: text" and mapping is cand_id->text.
    """
    indices = list(range(1, len(raw_candidates) + 1))
    random.shuffle(indices)
    lines: List[str] = []
    id_to_text: Dict[str, str] = {}
    for i, text in zip(indices, raw_candidates):
        cand_id = CAND_TOKEN_TEMPLATE.format(idx=i)
        lines.append(f"{cand_id}: {text}")
        id_to_text[cand_id] = text
    return lines, id_to_text


def build_pointer_list_target(
    gold_cand_id: str,
    ranked_other_ids: List[str],
    max_ranks: int = R_MAX_RANKS,
) -> List[str]:
    """
    Construct target lines for the pointer list section using <Rk>=<CAND_j> format.
    Truncates to at most max_ranks entries.
    """
    order = [gold_cand_id] + [cid for cid in ranked_other_ids if cid != gold_cand_id]
    order = order[:max_ranks]
    lines = []
    for k, cid in enumerate(order, start=1):
        lines.append(f"{RANK_TOKEN_TEMPLATE.format(rank=k)}={cid}")
    lines.append(ENDLIST_TOKEN)
    return lines


def format_reranker_prompt(
    question: str,
    asr_transcript: str,
    candidate_texts: List[str],
    reasoning_hint: Optional[str] = None,
) -> str:
    """
    Create the input text to condition the model. Video frames are supplied separately to the model.
    """
    cand_lines, _ = enumerate_and_shuffle_candidates(candidate_texts)
    parts: List[str] = []
    parts.append(f"Question: {question}")
    if asr_transcript:
        parts.append(f"ASR: {asr_transcript}")
    parts.append("Candidates:")
    parts.extend(cand_lines)
    parts.append("")
    parts.append(f"{REASONING_PREFIX} {reasoning_hint or ''}".rstrip())
    parts.append(FINAL_LIST_HEADER)
    return "\n".join(parts)



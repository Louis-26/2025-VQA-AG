from typing import List, Tuple, Dict, Optional

from .constants import (
    M_MAX_CANDIDATES,
    R_MAX_RANKS,
    RANK_TOKEN_TEMPLATE,
    CAND_TOKEN_TEMPLATE,
    ENDLIST_TOKEN,
)


def build_special_tokens(
    max_candidates: int = M_MAX_CANDIDATES,
    max_ranks: int = R_MAX_RANKS,
) -> List[str]:
    tokens: List[str] = []
    tokens.extend([RANK_TOKEN_TEMPLATE.format(rank=i) for i in range(1, max_ranks + 1)])
    tokens.extend([CAND_TOKEN_TEMPLATE.format(idx=i) for i in range(1, max_candidates + 1)])
    tokens.append(ENDLIST_TOKEN)
    return tokens


def add_special_tokens_to_tokenizer(
    tokenizer,
    model: Optional[object] = None,
    max_candidates: int = M_MAX_CANDIDATES,
    max_ranks: int = R_MAX_RANKS,
) -> int:
    """
    Adds rank and candidate pointer tokens (and <ENDLIST>) to the tokenizer.
    If a model is provided, resizes its token embeddings accordingly.

    Returns the number of tokens newly added.
    """
    new_tokens = build_special_tokens(max_candidates=max_candidates, max_ranks=max_ranks)
    unique_new_tokens = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": unique_new_tokens})
    if model is not None and num_added > 0:
        # Works for HF models that implement resize_token_embeddings
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            # Some engines (e.g., vLLM) don't expose this; ignore gracefully.
            pass
    return num_added


def get_rank_tokens(max_ranks: int = R_MAX_RANKS) -> List[str]:
    return [RANK_TOKEN_TEMPLATE.format(rank=i) for i in range(1, max_ranks + 1)]


def get_candidate_tokens(max_candidates: int = M_MAX_CANDIDATES) -> List[str]:
    return [CAND_TOKEN_TEMPLATE.format(idx=i) for i in range(1, max_candidates + 1)]



from typing import List, Set, Tuple

import torch


def _candidate_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    vocab = tokenizer.get_vocab()
    # Collect all tokens that match the <CAND_*> pattern present in tokenizer
    for tok, tid in vocab.items():
        if tok.startswith("<CAND_") and tok.endswith(">"):
            ids.append(tid)
    return sorted(ids)


def _endlist_token_id(tokenizer) -> int:
    return tokenizer.convert_tokens_to_ids("<ENDLIST>")


def _is_token(tokenizer, token_id: int, token_str: str) -> bool:
    return token_id == tokenizer.convert_tokens_to_ids(token_str)


def mask_logits_for_pointer_training(
    logits: torch.Tensor,          # (b, t, v)
    labels: torch.Tensor,          # (b, t)
    input_ids: torch.Tensor,       # (b, t)
    tokenizer,
) -> torch.Tensor:
    """
    Apply training-time vocabulary constraints for pointer list decoding.

    Logic:
    - At positions where label is a candidate ID token, restrict logits to the set of
      remaining candidate IDs that appear in the example (from input_ids+labels)
      minus those already selected earlier in the target.
    - At positions where label is <ENDLIST>, restrict logits to <ENDLIST> only.
    - Else, leave logits unchanged.
    """
    bsz, seqlen, vocab = logits.shape
    device = logits.device

    cand_ids_all: List[int] = _candidate_token_ids(tokenizer)
    endlist_id: int = _endlist_token_id(tokenizer)

    masked_logits = logits.clone()

    for b in range(bsz):
        # Candidate ids present in this example (prompt or target)
        present_cands: Set[int] = set()
        for tid in torch.unique(torch.cat([input_ids[b], labels[b]])).tolist():
            if tid in cand_ids_all:
                present_cands.add(tid)

        chosen_cands: Set[int] = set()

        for t in range(seqlen):
            y = labels[b, t].item()
            if y == -100:
                # ignore position
                continue
            if y == endlist_id:
                # Only allow ENDLIST at its position
                allowed = torch.tensor([endlist_id], device=device, dtype=torch.long)
                masked_logits[b, t, :] = float("-inf")
                masked_logits[b, t, allowed] = logits[b, t, allowed]
                continue

            if y in cand_ids_all:
                # Build remaining set: present - chosen
                remaining = sorted(list(present_cands.difference(chosen_cands)))
                if not remaining:
                    continue
                allowed = torch.tensor(remaining, device=device, dtype=torch.long)
                masked_logits[b, t, :] = float("-inf")
                masked_logits[b, t, allowed] = logits[b, t, allowed]
                # Mark this candidate as chosen (advance state)
                chosen_cands.add(y)
                continue

            # Otherwise, leave unmasked

    return masked_logits



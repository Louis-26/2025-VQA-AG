import re
from typing import List, Tuple

import torch

from .constants import RANK_TOKEN_TEMPLATE


def extract_present_candidate_tokens_from_prompt(prompt: str) -> List[str]:
    # Lines like "<CAND_3>: text"
    cand_ids = []
    for line in prompt.splitlines():
        m = re.match(r"^(<CAND_\d+>):\s", line.strip())
        if m:
            cand_ids.append(m.group(1))
    return cand_ids


def greedy_pointer_decode(
    model,
    tokenizer,
    prompt: str,
    max_ranks: int,
) -> List[str]:
    """
    Deterministically decode the pointer list by forcing the scaffold tokens and
    selecting argmax among remaining <CAND_*> tokens at each rank.

    Returns the list of predicted candidate token strings, e.g., ["<CAND_3>", ...].
    """
    device = next(model.parameters()).device
    present_cands = extract_present_candidate_tokens_from_prompt(prompt)
    present_ids = [tokenizer.convert_tokens_to_ids(c) for c in present_cands]
    chosen: List[int] = []
    emitted: List[str] = []

    scaffold = prompt

    for k in range(1, max_ranks + 1):
        rank_tok = RANK_TOKEN_TEMPLATE.format(rank=k)
        scaffold_k = f"{scaffold}\n{rank_tok}="
        input_ids = tokenizer(scaffold_k, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids)
            logits = out.logits[:, -1, :]  # (1, vocab)
        remaining = [tid for tid in present_ids if tid not in chosen]
        if not remaining:
            break
        scores = logits[:, remaining]  # (1, R)
        idx = torch.argmax(scores, dim=-1).item()
        next_tid = remaining[idx]
        chosen.append(next_tid)
        cand_token_str = tokenizer.convert_ids_to_tokens(next_tid)
        emitted.append(cand_token_str)
        # Append the chosen and a newline to scaffold for next rank
        scaffold = scaffold_k + cand_token_str + "\n"
    return emitted



from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_rank_weight_mask(
    input_ids: torch.Tensor,
    tokenizer,
    rank_weights: Dict[int, float],
) -> torch.Tensor:
    """
    Build a per-token weight mask aligned with target tokens, based on which
    rank pointer (<Rk>=<CAND_*>) the token belongs to.

    Args:
        input_ids: (batch, seq_len) target token ids (right-shifted labels or labels directly)
        tokenizer: HF tokenizer with special tokens for <Rk> and <CAND_i>
        rank_weights: map rank k -> weight (e.g., {1:3.0, 2:1.5})

    Returns:
        weights: (batch, seq_len) tensor of floats
    """
    bsz, seqlen = input_ids.shape
    weights = input_ids.new_ones((bsz, seqlen), dtype=torch.float32)

    # Gather token ids for rank markers and candidate ids
    rank_token_to_id = {}
    for k, w in rank_weights.items():
        tok = f"<R{k}>"
        if tok in tokenizer.get_vocab():
            rank_token_to_id[k] = tokenizer.convert_tokens_to_ids(tok)

    # We mark spans after seeing <Rk> until the end of that line (naive heuristic: until newline token or next <R*>).
    # For simplicity, we apply the weight to tokens in the same line as <Rk>.
    for k, tok_id in rank_token_to_id.items():
        mask = input_ids == tok_id  # (b, t)
        # Extend weight to a short window ahead to cover '= <CAND_*>' tokens; use fixed window of, say, 6 tokens
        for shift in range(0, 6):
            shifted = F.pad(mask[:, :-shift] if shift > 0 else mask, (shift, 0), value=False)
            weights = torch.where(shifted, torch.full_like(weights, float(rank_weights[k])), weights)
    return weights


def rank_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    rank_weights: Optional[Dict[int, float]] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Standard token-level CE with per-token weights for rank pointer lines.
    """
    # Compute base loss per token
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab), labels.view(-1), reduction="none", ignore_index=ignore_index
    ).view_as(labels)

    if rank_weights:
        weights = compute_rank_weight_mask(labels, tokenizer, rank_weights)
        loss = loss * weights

    # Mask out ignored positions
    valid = labels.ne(ignore_index)
    loss = (loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return loss


def plackett_luce_pointer_loss(
    pointer_logits: List[torch.Tensor],
    target_cand_ids: List[torch.Tensor],
    candidate_vocab_ids: List[List[int]],
) -> torch.Tensor:
    """
    Compute listwise PL loss over pointer steps.

    Args:
        pointer_logits: list length K, each tensor (batch, vocab) logits at the position of <Rk>=<CAND_*> token.
        target_cand_ids: list length K, each (batch,) of target candidate token ids (single-token IDs).
        candidate_vocab_ids: list length K, each is a Python list of remaining candidate token ids allowed at rank k.

    Returns:
        Scalar loss (averaged over batch and K).
    """
    losses = []
    for k, (logits_k, tgt_ids_k, cand_ids_k) in enumerate(zip(pointer_logits, target_cand_ids, candidate_vocab_ids)):
        # Mask logits to allowed candidates only
        mask = torch.full_like(logits_k, fill_value=float("-inf"))
        idx = torch.tensor(cand_ids_k, device=logits_k.device, dtype=torch.long)
        mask.index_copy_(1, idx, logits_k.index_select(1, idx))
        log_probs = F.log_softmax(mask, dim=-1)
        # Gather target log prob
        tgt_logp = log_probs.gather(1, tgt_ids_k.view(-1, 1)).squeeze(1)
        losses.append(-tgt_logp)
    loss = torch.stack(losses, dim=0).mean()
    return loss


def masked_pointer_ce_with_rank_weights(
    logits: torch.Tensor,           # (b, t, v)
    labels: torch.Tensor,           # (b, t)
    input_ids: torch.Tensor,        # (b, t)
    tokenizer,
    rank_weights: Optional[Dict[int, float]] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute CE with vocabulary restricted at pointer positions (candidate or ENDLIST)
    and apply rank-dependent weights on the candidate-id token at each rank.

    This avoids in-place masked logits and preserves gradient flow.
    """
    device = logits.device
    bsz, seqlen, vocab = logits.shape
    rank_weights = rank_weights or {}

    # Precompute id sets
    cand_ids_all: List[int] = [tid for tok, tid in tokenizer.get_vocab().items() if tok.startswith("<CAND_") and tok.endswith(">")]
    cand_ids_all = sorted(cand_ids_all)
    endlist_id: int = tokenizer.convert_tokens_to_ids("<ENDLIST>")

    total_loss = torch.zeros((), device=device)
    total_weight = torch.zeros((), device=device)

    for b in range(bsz):
        # Candidate ids present in this example (from prompt+target)
        present = set()
        for tid in torch.unique(torch.cat([input_ids[b], labels[b]])).tolist():
            if tid in cand_ids_all:
                present.add(tid)
        chosen = []  # maintain order to infer rank index

        for t in range(seqlen):
            y = labels[b, t].item()
            if y == ignore_index:
                continue

            logits_bt = logits[b, t]

            if y == endlist_id:
                log_probs = F.log_softmax(logits_bt, dim=-1)
                loss_bt = -log_probs[endlist_id]
                w = 1.0
            elif y in cand_ids_all:
                # Remaining set
                remaining = sorted(list(set(present).difference(set(chosen))))
                idx = torch.tensor(remaining, device=device, dtype=torch.long)
                # logsumexp over remaining
                lse = torch.logsumexp(logits_bt.index_select(0, idx), dim=-1)
                loss_bt = -(logits_bt[y] - lse)
                # Rank index = next position
                rank_idx = len(chosen) + 1
                w = float(rank_weights.get(rank_idx, 1.0))
                chosen.append(y)
            else:
                # Standard CE over full vocab
                loss_bt = F.cross_entropy(logits_bt.view(1, -1), torch.tensor([y], device=device), reduction="sum")
                w = 1.0

            total_loss = total_loss + loss_bt * w
            total_weight = total_weight + torch.tensor(w, device=device)

    return total_loss / total_weight.clamp_min(1.0)



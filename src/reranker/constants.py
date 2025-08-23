M_MAX_CANDIDATES: int = 16
R_MAX_RANKS: int = 10

# Special token string templates
RANK_TOKEN_TEMPLATE: str = "<R{rank}>"
CAND_TOKEN_TEMPLATE: str = "<CAND_{idx}>"
ENDLIST_TOKEN: str = "<ENDLIST>"

# Optional helper text markers used in prompts (kept as normal tokens)
REASONING_PREFIX: str = "Reasoning:"
FINAL_LIST_HEADER: str = "Final list:"



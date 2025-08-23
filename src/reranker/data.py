import json
from typing import List, Dict, Any, Optional

from torch.utils.data import Dataset


class RerankerJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str) -> None:
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        return {
            "prompt": ex.get("prompt", ""),
            "target": ex.get("target", ""),
        }


class PointerListCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [f["prompt"] for f in features]
        targets = [f["target"] for f in features]

        # Tokenize prompt and target separately to build labels with prompt masked out
        enc_prompt = self.tokenizer(
            prompts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        enc_target = self.tokenizer(
            targets,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        attention_batch: List[List[int]] = []

        for ids_p, ids_t, attn_p in zip(enc_prompt["input_ids"], enc_target["input_ids"], enc_prompt["attention_mask"]):
            input_ids = ids_p + ids_t
            labels = [-100] * len(ids_p) + ids_t
            attention = attn_p + [1] * len(ids_t)
            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_batch.append(attention)

        batch = self.tokenizer.pad(
            {
                "input_ids": input_ids_batch,
                "labels": labels_batch,
                "attention_mask": attention_batch,
            },
            padding=True,
            return_tensors="pt",
        )
        return batch



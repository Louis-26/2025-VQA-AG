import json
import os
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
            "Q_ID": ex.get("Q_ID"),
            "Video_ID": ex.get("Video_ID"),
            "prompt": ex.get("prompt", ""),
            "target": ex.get("target", ""),
        }


class PointerListCollator:
    def __init__(self, tokenizer, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [f["prompt"] for f in features]
        targets = [f["target"] for f in features]

        # Tokenize prompt and target separately to build labels with prompt masked out
        enc_prompt = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None,
            max_length=self.max_length,
        )
        enc_target = self.tokenizer(
            targets,
            padding=False,
            truncation=True,
            add_special_tokens=False,
            return_tensors=None,
            max_length=self.max_length,
        )

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        attention_batch: List[List[int]] = []

        for ids_p, ids_t, attn_p in zip(enc_prompt["input_ids"], enc_target["input_ids"], enc_prompt["attention_mask"]):
            input_ids = (ids_p + ids_t)[: self.max_length]
            cut = self.max_length - len(ids_p)
            cut = max(0, min(cut, len(ids_t)))
            labels = [-100] * len(ids_p) + ids_t[:cut]
            attention = attn_p + [1] * cut
            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_batch.append(attention)

        # Manually pad to uniform length (tokenizer.pad won't pad labels reliably)
        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(x) for x in input_ids_batch) if input_ids_batch else 0
        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids_batch = [pad_to(seq, pad_id, max_len) for seq in input_ids_batch]
        attention_batch = [pad_to(seq, 0, max_len) for seq in attention_batch]
        labels_batch = [pad_to(seq, -100, max_len) for seq in labels_batch]

        import torch
        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


class VideoPointerCollator:
    """Builds Qwen2.5-VL style inputs: videos + text prompt; labels cover only target.

    Expects each feature to include Q_ID, Video_ID, prompt, target and uses
    `videos_dir` to locate `<Video_ID>.mp4`. Samples `num_frames` evenly.
    """
    def __init__(self, processor, videos_dir: str, num_frames: int = 64, frame_size: Optional[int] = None) -> None:
        self.processor = processor
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts: List[str] = []
        video_inputs: List[Any] = []  # file URIs or numpy arrays (fallback)
        targets: List[str] = []

        # Prefer native decoding via processor by passing file:// URIs
        for f in features:
            vid = f.get("Video_ID")
            video_path = f"{self.videos_dir}/{vid}.mp4"
            if os.path.exists(video_path):
                video_uri = f"file://{os.path.abspath(video_path)}"
                video_inputs.append(video_uri)
                texts.append(f["prompt"])  # prompt contains Question/ASR/Candidates/headers
                targets.append(f["target"])  # pointer lines

        # If no valid files found, keep training alive with a 1-frame dummy
        if len(video_inputs) == 0:
            import numpy as np
            dummy = np.zeros((1, 224, 224, 3), dtype=np.uint8)
            video_inputs = [dummy]
            texts = [features[0]["prompt"]]
            targets = [features[0]["target"]]

        # Build chat template style inputs
        messages = []
        for text, v in zip(texts, video_inputs):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": v},
                    {"type": "text", "text": text},
                ],
            })

        # Convert to model inputs (processor will decode if given file URIs)
        input_texts = [self.processor.apply_chat_template([m], tokenize=False, add_generation_prompt=False) for m in messages]
        model_inputs = self.processor(text=input_texts, videos=video_inputs, return_tensors="pt", padding=True)

        # Build labels by tokenizing targets separately and masking inputs length
        target_enc = self.processor.tokenizer(
            targets, padding=True, truncation=False, add_special_tokens=False, return_tensors="pt"
        )
        input_len = model_inputs["input_ids"].size(1)
        max_tgt = target_enc["input_ids"].size(1)
        batch_kept = len(video_inputs)
        padded_labels = target_enc["input_ids"].new_full((batch_kept, input_len + max_tgt), -100)
        # copy input ids into the front (ignored by -100 labels)
        padded_inputs = model_inputs["input_ids"].new_zeros((batch_kept, input_len + max_tgt))
        padded_inputs[:, :input_len] = model_inputs["input_ids"]
        # place target ids after inputs
        padded_inputs[:, input_len:input_len + max_tgt] = target_enc["input_ids"]
        labels = padded_labels.clone()
        labels[:, input_len:input_len + max_tgt] = target_enc["input_ids"]

        # Replace input_ids and attention to reflect concatenation
        attention = padded_inputs.ne(0).long()
        model_inputs["input_ids"] = padded_inputs
        model_inputs["attention_mask"] = attention
        model_inputs["labels"] = labels
        return model_inputs



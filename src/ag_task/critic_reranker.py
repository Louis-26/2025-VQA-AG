import re
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


class CriticImportError(RuntimeError):
    pass


def _safe_import_llava():
    try:
        from llava.model.builder import load_pretrained_model  # type: ignore
        from llava.mm_utils import process_images, tokenizer_image_token  # type: ignore
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
        from llava.conversation import conv_templates  # type: ignore
        return {
            "load_pretrained_model": load_pretrained_model,
            "process_images": process_images,
            "tokenizer_image_token": tokenizer_image_token,
            "IMAGE_TOKEN_INDEX": IMAGE_TOKEN_INDEX,
            "DEFAULT_IMAGE_TOKEN": DEFAULT_IMAGE_TOKEN,
            "conv_templates": conv_templates,
        }
    except Exception as e:
        raise CriticImportError(
            "LLaVA-Critic dependencies not found. Install LLaVA-NeXT:\n"
            "  pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
        ) from e


def _to_pil_frames(frames: np.ndarray) -> List[Image.Image]:
    frames = np.asarray(frames)
    pil_list: List[Image.Image] = []
    for f in frames:
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        pil_list.append(Image.fromarray(f[..., ::-1]) if f.shape[-1] == 3 else Image.fromarray(f))
    return pil_list


def _sample_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return list(np.linspace(0, n - 1, k, dtype=int))


class LlavaCriticReranker:
    def __init__(
        self,
        model_name: str = "lmms-lab/llava-critic-7b",
        device_map: str = "auto",
        max_images: int = 8,
        conv_template: str = "qwen_1_5",
    ) -> None:
        mods = _safe_import_llava()
        self._load_pretrained_model = mods["load_pretrained_model"]
        self._process_images = mods["process_images"]
        self._tokenizer_image_token = mods["tokenizer_image_token"]
        self._IMAGE_TOKEN_INDEX = mods["IMAGE_TOKEN_INDEX"]
        self._DEFAULT_IMAGE_TOKEN = mods["DEFAULT_IMAGE_TOKEN"]
        self._conv_templates = mods["conv_templates"]

        self.model_name = model_name
        self.device_map = device_map
        self.max_images = max_images
        self.conv_template = conv_template

        # Ensure tokenizer resize works across transformer versions (dict-based text_config)
        try:
            from transformers.modeling_utils import PreTrainedModel  # type: ignore

            _orig_resize = PreTrainedModel.resize_token_embeddings

            def _safe_resize_token_embeddings(self_model, new_num_tokens=None, pad_to_multiple_of=None, *args, **kwargs):
                text_cfg = getattr(self_model.config, "text_config", None)
                if isinstance(text_cfg, dict):
                    class _CfgObj:
                        pass
                    obj = _CfgObj()
                    for k, v in text_cfg.items():
                        setattr(obj, k, v)
                    self_model.config.text_config = obj
                return _orig_resize(self_model, new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, *args, **kwargs)

            PreTrainedModel.resize_token_embeddings = _safe_resize_token_embeddings  # type: ignore
        except Exception:
            pass

        # load model
        tokenizer, model, image_processor, max_length = self._load_pretrained_model(
            model_path=model_name,
            model_base=None,
            model_name="llava_qwen",
            device_map=device_map,
            attn_implementation="sdpa",
        )
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.image_processor = image_processor
        self.max_length = max_length

    def _build_prompt(self, question: str, answer: str, transcript: Optional[str]) -> str:
        rubric = (
            "You are an unbiased visual judge. Evaluate the answer grounded in frames and transcript.\n"
            "Criteria: factual grounding, temporal correctness, audio/transcript alignment, specificity/conciseness.\n"
            "Output strictly in this format: 'Score: X/10\nReason: <one short paragraph>' where X is an integer 0-10."
        )
        parts = [rubric, f"Question: {question}", f"Answer: {answer}"]
        if transcript:
            parts.append(f"Transcript: {transcript}")
        return "\n".join(parts)

    def _parse_score(self, text: str) -> Optional[float]:
        m = re.search(r"Score:\s*(\d+(?:\.\d+)?)/10", text)
        if m:
            try:
                v = float(m.group(1))
                v = max(0.0, min(10.0, v))
                return v
            except Exception:
                return None
        # fallback: any number 0-10
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
        if m2:
            try:
                v = float(m2.group(1))
                v = max(0.0, min(10.0, v))
                return v
            except Exception:
                return None
        return None

    def score_candidates(
        self,
        frames: np.ndarray,
        question: str,
        candidates: List[str],
        transcript: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> List[Tuple[str, float, float, str]]:
        """
        Returns list of tuples: (answer, score_0_10, latency_sec, raw_text)
        """
        if frames.size == 0:
            return [(c, 0.0, 0.0, "") for c in candidates]

        idxs = _sample_indices(frames.shape[0], self.max_images)
        sampled = frames[idxs]
        pil_images = _to_pil_frames(sampled)

        image_tensor = self._process_images(pil_images, self.image_processor, self.model.config)
        # Ensure dtype/device match the model (avoid float32 vs fp16/bf16 mismatch)
        target_dtype = getattr(self.model, "dtype", torch.float16)
        image_tensor = image_tensor.to(self.model.device, dtype=target_dtype)
        image_sizes = [im.size for im in pil_images]

        results: List[Tuple[str, float, float, str]] = []
        for ans in candidates:
            judge_prompt = self._build_prompt(question, ans, transcript)
            question_with_img = self._DEFAULT_IMAGE_TOKEN + "\n" + judge_prompt
            conv = self._conv_templates[self.conv_template].copy()
            conv = self._conv_templates[self.conv_template]
            conv = conv.copy()
            conv.append_message(conv.roles[0], question_with_img)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self._tokenizer_image_token(
                prompt, self.tokenizer, self._IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            st = time.time()
            with np.errstate(all='ignore'):
                cont = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=(temperature > 0.0),
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            latency = time.time() - st
            text_out = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            print(text_out)
            score = self._parse_score(text_out) or 0.0
            results.append((ans, float(score), float(latency), text_out))
        return results

    def rerank(
        self,
        frames: np.ndarray,
        question: str,
        candidates: List[str],
        transcript: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        scored = self.score_candidates(frames, question, candidates, transcript)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(a, s) for (a, s, _, _) in scored]

import argparse
from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from src.reranker.data import RerankerJsonlDataset, PointerListCollator
from src.reranker.tokens import add_special_tokens_to_tokenizer
from src.reranker.losses import masked_pointer_ce_with_rank_weights


@dataclass
class RankWeights:
    w1: float = 3.0
    w2: float = 1.5
    w3: float = 1.2

    def to_dict(self) -> Dict[int, float]:
        return {1: self.w1, 2: self.w2, 3: self.w3}


class MaskedWeightedTrainer(Trainer):
    def __init__(self, rank_weights: Dict[int, float], tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_weights = rank_weights
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss = masked_pointer_ce_with_rank_weights(
            logits=logits,
            labels=labels,
            input_ids=inputs["input_ids"],
            tokenizer=self.tokenizer,
            rank_weights=self.rank_weights,
        )
        return (loss, outputs) if return_outputs else loss


def main():
    p = argparse.ArgumentParser("Train Qwen reranker with masked, rank-weighted SFT")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=False)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    # Rank weights
    p.add_argument("--w1", type=float, default=3.0, help="Rank 1 weight")
    p.add_argument("--w2", type=float, default=1.5, help="Rank 2 weight")
    p.add_argument("--w3", type=float, default=1.2, help="Rank 3 weight")
    # LoRA options
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module name fragments for LoRA",
    )
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(cfg, "model_type", "") == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True
        )
    add_special_tokens_to_tokenizer(tokenizer, model)

    # LoRA wrapping (reduces memory use)
    if args.use_lora:
        target_modules = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        # Optional: show trainable parameter count
        try:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"LoRA enabled. Trainable params: {trainable:,} / {total:,}")
        except Exception:
            pass

    # Memory savers
    try:
        # Use non-reentrant to avoid requiring inputs with requires_grad=True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    # Ensure inputs can require grad for checkpointing paths
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception:
        pass
    try:
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass

    train_ds = RerankerJsonlDataset(args.train_jsonl)
    data_collator = PointerListCollator(tokenizer)

    rank_weights = {1: args.w1, 2: args.w2, 3: args.w3}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=50,
        save_steps=1000,
        gradient_accumulation_steps=1,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = MaskedWeightedTrainer(
        rank_weights=rank_weights,
        tokenizer=tokenizer,
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()



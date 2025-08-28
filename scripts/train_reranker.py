import argparse
from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments
import os
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from src.reranker.data import RerankerJsonlDataset, PointerListCollator, VideoPointerCollator
from src.reranker.tokens import add_special_tokens_to_tokenizer
from src.reranker.losses import masked_pointer_ce_with_rank_weights

# Optional plotting of training loss vs epoch
import json
import glob
import os

def plot_training_loss(output_dir: str, num_epochs: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Warning: matplotlib not available; skipping loss plot.")
        return
    loss_data = []
    epoch_data = []
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                if "loss" in entry and "epoch" in entry:
                    loss_data.append(entry["loss"])
                    epoch_data.append(entry["epoch"])
        except Exception:
            pass
    for cp in sorted(glob.glob(os.path.join(output_dir, "checkpoint-*"))):
        cp_state = os.path.join(cp, "trainer_state.json")
        if not os.path.exists(cp_state):
            continue
        try:
            with open(cp_state, "r") as f:
                s = json.load(f)
            e = s.get("epoch")
            if e is None:
                continue
            for entry in reversed(s.get("log_history", [])):
                if "loss" in entry:
                    if e not in epoch_data:
                        loss_data.append(entry["loss"])
                        epoch_data.append(e)
                    break
        except Exception:
            pass
    if not loss_data:
        print("Warning: No loss data found to plot.")
        return
    pairs = sorted(zip(epoch_data, loss_data), key=lambda x: x[0])
    epoch_data, loss_data = zip(*pairs)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_data, loss_data, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reranker Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, "training_loss_plot.png")
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        print(f"Saved loss plot to {plot_path}")
    except Exception as e:
        print(f"Warning: Failed to save loss plot: {e}")


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
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_accum", type=int, default=1)
    # Video options
    p.add_argument("--use_video", action="store_true")
    p.add_argument("--videos_dir", type=str, default="/brtx/603-nvme1/yweng13/VQA/my_train_videos")
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--frame_size", type=int, default=None, help="If set, resize frames to SxS (e.g., 224)")
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
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    # Avoid device_map="auto" under DDP; let Accelerator move the model
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_device_map = None if world_size > 1 else "auto"
    if getattr(cfg, "model_type", "") == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, device_map=use_device_map, trust_remote_code=True
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map=use_device_map, trust_remote_code=True
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

    # Memory savers - but avoid gradient checkpointing under DDP due to LoRA conflicts
    enable_grad_ckpt = world_size == 1  # Only enable for single-GPU training
    if enable_grad_ckpt:
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
    if args.use_video:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        data_collator = VideoPointerCollator(
            processor,
            videos_dir=args.videos_dir,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
        )
    else:
        data_collator = PointerListCollator(tokenizer)

    rank_weights = {1: args.w1, 2: args.w2, 3: args.w3}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        save_steps=1000,
        save_total_limit=5,
        gradient_accumulation_steps=args.grad_accum,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=enable_grad_ckpt,
        warmup_steps=args.warmup_steps,
        ddp_find_unused_parameters=True,
    )

    trainer = MaskedWeightedTrainer(
        rank_weights=rank_weights,
        tokenizer=tokenizer,
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    # Plot loss vs epoch
    try:
        plot_training_loss(args.output_dir, num_epochs=args.epochs)
    except Exception as e:
        print(f"Warning: could not plot loss: {e}")


if __name__ == "__main__":
    main()



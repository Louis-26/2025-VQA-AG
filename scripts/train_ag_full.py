#!/usr/bin/env python3
"""
Full fine-tuning script for Answer Generation (AG) model.

Trains Qwen2.5-VL-7B-Instruct to generate 10 answers and maximize BERTScore with ground truth.
Uses full parameter updates (no LoRA).
"""
import argparse
import os
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from transformers import (
    AutoTokenizer, AutoProcessor, AutoConfig,
    Trainer, TrainingArguments,
    set_seed
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from src.ag_task.dataset import AGTrainingDataset
from src.ag_task.collators import AGVideoCollator, AGTextCollator
from src.ag_task.losses import BERTScoreMaxLoss, SimpleBERTScoreMaxLoss


class AGTrainer(Trainer):
    """
    Custom trainer for AG model with BERTScore-based loss.
    """
    
    def __init__(
        self, 
        loss_fn,
        use_bertscore: bool = True,
        num_answers: int = 10,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.use_bertscore = use_bertscore
        self.num_answers = num_answers
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute custom BERTScore-based loss.
        """
        if not self.use_bertscore:
            # Fallback to standard cross-entropy loss
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k not in ["labels", "ground_truths", "q_ids"]})
            logits = outputs.logits
            
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = torch.tensor(0.0, device=logits.device)
            
            return (loss, outputs) if return_outputs else loss
        
        # BERTScore-based training
        ground_truths = inputs.get("ground_truths", [])
        if not ground_truths:
            # Fallback if no ground truths available
            return self.compute_loss(model, inputs, return_outputs, num_items_in_batch)
        
        # Generate answers from the model
        model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "ground_truths", "q_ids"]}
        
        with torch.no_grad():
            # Generate multiple sequences
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode generated text
            input_length = model_inputs["input_ids"].size(1)
            generated_texts = []
            for i, ids in enumerate(generated_ids):
                # Remove input tokens to get only generated part
                generated_only = ids[input_length:]
                text = self.tokenizer.decode(generated_only, skip_special_tokens=True)
                generated_texts.append(text)
        
        # Compute BERTScore loss
        if len(generated_texts) == len(ground_truths):
            loss = self.loss_fn(generated_texts, ground_truths)
        else:
            # Fallback loss if batch sizes don't match
            loss = torch.tensor(0.0, device=model_inputs["input_ids"].device, requires_grad=True)
        
        if return_outputs:
            # Create dummy outputs for compatibility
            outputs = type('Outputs', (), {})()
            outputs.loss = loss
            return loss, outputs
        
        return loss


def main():
    parser = argparse.ArgumentParser("Full fine-tuning for AG model")
    
    # Model and data arguments
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model name")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file")
    parser.add_argument("--val_jsonl", help="Validation JSONL file")
    parser.add_argument("--videos_dir", default="/brtx/603-nvme1/yweng13/VQA/my_train_videos", help="Video directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging every N steps")
    
    # Model-specific arguments
    parser.add_argument("--use_video", action="store_true", help="Use video input")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of video frames")
    parser.add_argument("--num_answers", type=int, default=10, help="Number of answers to generate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Loss function arguments
    parser.add_argument("--use_bertscore", action="store_true", help="Use BERTScore loss (experimental)")
    parser.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli", help="BERTScore model")
    parser.add_argument("--simple_bertscore", action="store_true", help="Use simplified BERTScore loss")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    
    # Load model
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device_map = None if world_size > 1 else "auto"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Ensure model can be trained
    for param in model.parameters():
        param.requires_grad = True
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Memory optimizations
    if hasattr(model, "gradient_checkpointing_enable") and world_size == 1:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
    
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    # Load datasets
    print(f"Loading training data: {args.train_jsonl}")
    train_dataset = AGTrainingDataset(args.train_jsonl, args.videos_dir)
    
    val_dataset = None
    if args.val_jsonl:
        print(f"Loading validation data: {args.val_jsonl}")
        val_dataset = AGTrainingDataset(args.val_jsonl, args.videos_dir)
    
    # Setup data collator
    if args.use_video:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        data_collator = AGVideoCollator(
            processor=processor,
            videos_dir=args.videos_dir,
            num_answers=args.num_answers,
            num_frames=args.num_frames,
            max_length=args.max_length
        )
    else:
        data_collator = AGTextCollator(
            tokenizer=tokenizer,
            num_answers=args.num_answers,
            max_length=args.max_length
        )
    
    # Setup loss function
    loss_fn = None
    if args.use_bertscore:
        print(f"Using BERTScore loss with model: {args.bertscore_model}")
        if args.simple_bertscore:
            loss_fn = SimpleBERTScoreMaxLoss(
                num_answers=args.num_answers,
                bertscore_model=args.bertscore_model
            )
        else:
            loss_fn = BERTScoreMaxLoss(
                num_answers=args.num_answers,
                bertscore_model=args.bertscore_model
            )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=world_size == 1,  # Disable for DDP
        ddp_find_unused_parameters=True if world_size > 1 else False,
    )
    
    # Initialize trainer
    trainer = AGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        loss_fn=loss_fn,
        use_bertscore=args.use_bertscore,
        num_answers=args.num_answers
    )
    
    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()

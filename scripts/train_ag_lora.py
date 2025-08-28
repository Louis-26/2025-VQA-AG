#!/usr/bin/env python3
"""
LoRA fine-tuning script for Answer Generation (AG) model.

Trains Qwen2.5-VL-7B-Instruct with LoRA to generate 10 answers and maximize BERTScore with ground truth.
Memory-efficient alternative to full fine-tuning.
"""
import argparse
import os
import torch
import json
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from transformers import (
    AutoTokenizer, AutoProcessor, AutoConfig,
    Trainer, TrainingArguments,
    set_seed
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from src.ag_task.dataset import AGTrainingDataset
from src.ag_task.collators import AGVideoCollator, AGTextCollator
from src.ag_task.losses import BERTScoreMaxLoss, SimpleBERTScoreMaxLoss


class AGLoRATrainer(Trainer):
    """
    Custom trainer for AG model with LoRA.
    Uses standard LM loss for training and BERTScore for evaluation.
    """
    
    def __init__(
        self, 
        bertscore_evaluator=None,
        num_answers: int = 10,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bertscore_evaluator = bertscore_evaluator
        self.num_answers = num_answers
        self.generation_step_count = 0
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute standard LM loss for training. BERTScore is only used for evaluation.
        """
        # Always use standard cross-entropy loss for training
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
            
        # Optionally compute BERTScore for logging (every N steps to save compute)
        if (self.bertscore_evaluator is not None and 
            self.state.global_step % 50 == 0 and  # Only every 50 steps
            self.state.global_step > 0):
            self._log_bertscore_evaluation(model, inputs)
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_bertscore_evaluation(self, model, inputs):
        """
        Generate answers and compute BERTScore for evaluation (no gradients).
        """
        ground_truths = inputs.get("ground_truths", [])
        if not ground_truths:
            return
        
        try:
            with torch.no_grad():
                # Generate answers
                model_inputs = {k: v for k, v in inputs.items() if k not in ["labels", "ground_truths", "q_ids"]}
                
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
                for ids in generated_ids:
                    generated_only = ids[input_length:]
                    text = self.tokenizer.decode(generated_only, skip_special_tokens=True)
                    generated_texts.append(text)
                
                # Compute BERTScore
                if len(generated_texts) == len(ground_truths):
                    bertscore_result = self.bertscore_evaluator(generated_texts, ground_truths, return_scores=True)
                    if isinstance(bertscore_result, tuple):
                        _, bert_scores = bertscore_result
                        max_scores = torch.max(bert_scores, dim=1)[0]
                        avg_max_bertscore = max_scores.mean().item()
                        
                        # Log the metric
                        self.log({"eval/avg_max_bertscore": avg_max_bertscore})
                        
                        # Log a sample generation for inspection
                        if len(generated_texts) > 0:
                            sample_text = generated_texts[0][:200] + "..." if len(generated_texts[0]) > 200 else generated_texts[0]
                            print(f"\nStep {self.state.global_step} - Sample generation:")
                            print(f"Generated: {sample_text}")
                            print(f"Ground truth: {ground_truths[0]}")
                            print(f"Avg Max BERTScore: {avg_max_bertscore:.4f}")
                            
        except Exception as e:
            print(f"Warning: BERTScore evaluation failed: {e}")
            pass


def plot_training_loss(output_dir: str, num_epochs: int):
    """
    Plot training loss vs epoch from trainer state and checkpoint data.
    
    Args:
        output_dir: Directory containing training outputs and checkpoints
        num_epochs: Number of training epochs
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Collect loss data from trainer state and checkpoints
    loss_data = []
    epoch_data = []
    
    # Read main trainer state
    trainer_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
            
        # Extract from log history
        for entry in trainer_state.get("log_history", []):
            if "loss" in entry and "epoch" in entry:
                loss_data.append(entry["loss"])
                epoch_data.append(entry["epoch"])
    
    # Read checkpoint trainer states for additional data points
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for checkpoint_dir in sorted(checkpoint_dirs):
        checkpoint_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(checkpoint_state_path):
            with open(checkpoint_state_path, 'r') as f:
                checkpoint_state = json.load(f)
                
            # Get the final epoch for this checkpoint
            final_epoch = checkpoint_state.get("epoch")
            if final_epoch is not None:
                # Find the last loss value from this checkpoint's history
                log_history = checkpoint_state.get("log_history", [])
                if log_history:
                    for entry in reversed(log_history):
                        if "loss" in entry:
                            # Only add if we don't already have this epoch
                            if final_epoch not in epoch_data:
                                loss_data.append(entry["loss"])
                                epoch_data.append(final_epoch)
                            break
    
    if not loss_data:
        print("Warning: No loss data found for plotting")
        return
    
    # Sort by epoch
    combined = list(zip(epoch_data, loss_data))
    combined.sort(key=lambda x: x[0])
    epoch_data, loss_data = zip(*combined)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_data, loss_data, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('LoRA Training Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set epoch range
    plt.xlim(0, num_epochs)
    
    # Add loss values as text annotations
    for epoch, loss in zip(epoch_data, loss_data):
        plt.annotate(f'{loss:.4f}', 
                    (epoch, loss), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9,
                    alpha=0.8)
    
    # Save the plot
    plot_path = os.path.join(output_dir, "training_loss_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {plot_path}")
    
    # Also save loss data as CSV for further analysis
    csv_path = os.path.join(output_dir, "training_loss_data.csv")
    with open(csv_path, 'w') as f:
        f.write("epoch,loss\n")
        for epoch, loss in zip(epoch_data, loss_data):
            f.write(f"{epoch},{loss}\n")
    print(f"Training loss data saved to: {csv_path}")
    
    # Print summary statistics
    if len(loss_data) > 1:
        initial_loss = loss_data[0]
        final_loss = loss_data[-1]
        min_loss = min(loss_data)
        max_loss = max(loss_data)
        
        print(f"\nTraining Loss Summary:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Best loss: {min_loss:.4f}")
        print(f"  Worst loss: {max_loss:.4f}")
        print(f"  Improvement: {((initial_loss - final_loss) / initial_loss * 100):.2f}%")
        print(f"  Data points: {len(loss_data)}")


def main():
    parser = argparse.ArgumentParser("LoRA fine-tuning for AG model")
    
    # Model and data arguments
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model name")
    parser.add_argument("--train_jsonl", required=True, help="Training JSONL file")
    parser.add_argument("--val_jsonl", help="Validation JSONL file")
    parser.add_argument("--videos_dir", default="/brtx/603-nvme1/yweng13/VQA/my_train_videos", help="Video directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps (ignored if --save_every_epoch)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation every N steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging every N steps")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint after every epoch")
    parser.add_argument("--plot_loss", action="store_true", help="Plot loss vs epoch after training")
    
    # LoRA-specific arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules"
    )
    
    # Model-specific arguments
    parser.add_argument("--use_video", action="store_true", help="Use video input")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of video frames")
    parser.add_argument("--num_answers", type=int, default=10, help="Number of answers to generate")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    
    # Evaluation arguments
    parser.add_argument("--use_bertscore", action="store_true", help="Enable BERTScore evaluation during training (no gradient)")
    parser.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli", help="BERTScore model for evaluation")
    parser.add_argument("--simple_bertscore", action="store_true", help="Use simplified BERTScore evaluator")
    
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
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA
    print(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}")
    target_modules = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    try:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled. Trainable: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")
    except Exception:
        pass
    
    # Memory optimizations (avoid with DDP due to LoRA conflicts)
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
    
    # Setup BERTScore evaluator (for evaluation only, not training)
    bertscore_evaluator = None
    if args.use_bertscore:
        print(f"Using BERTScore for evaluation with model: {args.bertscore_model}")
        print("Note: BERTScore is used for evaluation only. Training uses standard LM loss.")
        if args.simple_bertscore:
            bertscore_evaluator = SimpleBERTScoreMaxLoss(
                num_answers=args.num_answers,
                bertscore_model=args.bertscore_model
            )
        else:
            bertscore_evaluator = BERTScoreMaxLoss(
                num_answers=args.num_answers,
                bertscore_model=args.bertscore_model
            )
    
    # Calculate steps per epoch for epoch-based saving
    dataset_size = len(train_dataset)
    steps_per_epoch = max(1, dataset_size // (args.batch_size * args.grad_accum * world_size))
    
    # Determine save strategy and steps
    if args.save_every_epoch:
        save_strategy = "epoch"
        save_steps = 1  # Save every 1 epoch
        save_total_limit = args.epochs + 1  # Keep all epoch checkpoints + final
        print(f"Epoch-based saving enabled: {steps_per_epoch} steps per epoch, saving every epoch")
    else:
        save_strategy = "steps"
        save_steps = args.save_steps
        save_total_limit = 3
        print(f"Step-based saving: every {save_steps} steps")
    
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
        save_steps=save_steps,
        eval_steps=args.eval_steps if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
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
    trainer = AGLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        bertscore_evaluator=bertscore_evaluator,
        num_answers=args.num_answers
    )
    
    # Train
    print("Starting LoRA training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print(f"Saving LoRA model to: {args.output_dir}")
    trainer.save_model()
    trainer.save_state()
    
    # Plot loss vs epoch if requested
    if args.plot_loss:
        try:
            plot_training_loss(args.output_dir, args.epochs)
        except Exception as e:
            print(f"Warning: Failed to plot training loss: {e}")


if __name__ == "__main__":
    main()

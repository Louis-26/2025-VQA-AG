"""
Data collators for Answer Generation model training.
"""
import os
from typing import List, Dict, Any, Optional
import torch


class AGVideoCollator:
    """
    Collator for AG model training with video input.
    
    Handles Qwen2.5-VL format for multimodal input with video frames.
    Creates prompts that instruct the model to generate 10 numbered answers.
    """
    
    def __init__(
        self, 
        processor, 
        videos_dir: str,
        num_answers: int = 10,
        num_frames: int = 32,
        max_length: int = 2048
    ):
        """
        Args:
            processor: Qwen2.5-VL processor for tokenization and video processing
            videos_dir: Directory containing video files
            num_answers: Number of answers to generate
            num_frames: Number of frames to sample from video
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.videos_dir = videos_dir
        self.num_answers = num_answers
        self.num_frames = num_frames
        self.max_length = max_length
    
    def create_ag_prompt(self, question: str, asr_transcript: str = "") -> str:
        """
        Create prompt for answer generation training.
        
        Instructs the model to generate exactly num_answers numbered answers.
        """
        prompt_parts = [
            "Answer the following question concisely in one sentence, you should follow the points:",
            "",
            "1. You should answer the question as simple as possible, some questions may just need a word or two.",
            "2. You don't need to answer the question in a very detailed way, just give a concise answer.",
            "3. For the answer in number, you should answer the number (one, two, three, etc.) but not 1, 2, 3, etc. in the question.",
        ]

        # Include transcript if provided
        prompt_parts.extend([
            "",
            "the transcript of the video is:",
        ])
        if asr_transcript.strip():
            prompt_parts.append(asr_transcript)

        # Question section
        prompt_parts.extend([
            "",
            "question:",
            question,
            "",
            f"Please provide exactly {self.num_answers} different possible answers to this question based on the video content.",
            f"Format your response as a numbered list from 1 to {self.num_answers}:",
            "",
        ])

        return "\n".join(prompt_parts)
    
    def create_target_answers(self, ground_truth: str) -> str:
        """
        Create target text with ground truth as first answer and variations/related answers.
        
        For training, we'll put the ground truth as answer #1 and generate variations.
        In practice, you might want to create more sophisticated targets.
        """
        target_lines = [
            f"1. {ground_truth}"
        ]
        
        # Add some variations of the ground truth for positions 2-10
        # In practice, you might want to use paraphrasing models or manual annotations
        for i in range(2, self.num_answers + 1):
            target_lines.append(f"{i}. {ground_truth}")  # Simple repetition for now
        
        return "\n".join(target_lines)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of AG training examples.
        
        Returns:
            Dictionary with input_ids, attention_mask, pixel_values, labels, and metadata
        """
        texts = []
        video_inputs = []
        targets = []
        valid_indices = []
        
        # Process each example
        for i, feature in enumerate(features):
            video_path = feature.get("video_path")
            
            # Skip examples without valid video
            if not video_path or not os.path.exists(video_path):
                continue
            
            # Create prompt
            prompt = self.create_ag_prompt(
                question=feature["question"],
                asr_transcript=feature.get("asr_transcript", "")
            )
            
            # Create target (ground truth formatted as numbered list)
            target = self.create_target_answers(feature["ground_truth"])
            
            # Add video file URI for processor
            video_uri = f"file://{os.path.abspath(video_path)}"
            
            texts.append(prompt)
            video_inputs.append(video_uri)
            targets.append(target)
            valid_indices.append(i)
        
        # Handle case where no valid videos found
        if not video_inputs:
            # Create a minimal fallback batch with text-only
            fallback_prompt = self.create_ag_prompt(
                question="What do you see?",
                asr_transcript=""
            )
            fallback_target = self.create_target_answers("I see a video.")
            
            # Process fallback as text-only
            model_inputs = self.processor(
                text=[fallback_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Process target
            target_encodings = self.processor.tokenizer(
                [fallback_target],
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_length
            )
            
            # Create labels for fallback
            input_len = model_inputs["input_ids"].shape[1]
            target_len = target_encodings["input_ids"].shape[1]
            
            full_input_ids = torch.cat([
                model_inputs["input_ids"],
                target_encodings["input_ids"]
            ], dim=1)
            
            full_attention = torch.cat([
                model_inputs["attention_mask"],
                target_encodings["attention_mask"]
            ], dim=1)
            
            labels = torch.full_like(full_input_ids, -100)
            labels[:, input_len:input_len + target_len] = target_encodings["input_ids"]
            
            model_inputs["input_ids"] = full_input_ids
            model_inputs["attention_mask"] = full_attention
            model_inputs["labels"] = labels
            model_inputs["ground_truths"] = ["I see a video."]
            model_inputs["q_ids"] = ["fallback"]
            
            return model_inputs
        
        # Build messages for chat template
        messages = []
        for text, video in zip(texts, video_inputs):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": text}
                ]
            })
        
        # Apply chat template and process
        input_texts = [
            self.processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        # Process inputs (tokenize + encode videos)
        # Note: videos are already embedded in input_texts via chat template, 
        # so we don't pass videos separately
        model_inputs = self.processor(
            text=input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Process targets
        target_encodings = self.processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length
        )
        
        # Create labels by concatenating inputs + targets, masking inputs
        input_len = model_inputs["input_ids"].size(1)
        target_len = target_encodings["input_ids"].size(1)
        batch_size = len(video_inputs)
        
        # Concatenate input and target tokens
        full_input_ids = torch.cat([
            model_inputs["input_ids"],
            target_encodings["input_ids"]
        ], dim=1)
        
        # Create attention mask
        full_attention = torch.cat([
            model_inputs["attention_mask"],
            target_encodings["attention_mask"]
        ], dim=1)
        
        # Create labels (mask input tokens with -100)
        labels = torch.full_like(full_input_ids, -100)
        labels[:, input_len:input_len + target_len] = target_encodings["input_ids"]
        
        # Update model inputs
        model_inputs["input_ids"] = full_input_ids
        model_inputs["attention_mask"] = full_attention
        model_inputs["labels"] = labels
        
        # Add metadata for loss computation
        model_inputs["ground_truths"] = [features[i]["ground_truth"] for i in valid_indices]
        model_inputs["q_ids"] = [features[i]["Q_ID"] for i in valid_indices]
        
        return model_inputs


class AGTextCollator:
    """
    Text-only collator for AG model training (without video).
    Useful for faster training or when video processing is not needed.
    """
    
    def __init__(
        self, 
        tokenizer, 
        num_answers: int = 10,
        max_length: int = 2048
    ):
        """
        Args:
            tokenizer: Tokenizer for text processing
            num_answers: Number of answers to generate
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.num_answers = num_answers
        self.max_length = max_length
    
    def create_ag_prompt(self, question: str, asr_transcript: str = "") -> str:
        """Create prompt for answer generation (text-only version)."""
        prompt_parts = [
            "Answer the following question concisely in one sentence, you should follow the points:",
            "",
            "1. You should answer the question as simple as possible, some questions may just need a word or two.",
            "2. You don't need to answer the question in a very detailed way, just give a concise answer.",
            "3. For the answer in number, you should answer the number (one, two, three, etc.) but not 1, 2, 3, etc. in the question.",
        ]

        # Include transcript if provided
        prompt_parts.extend([
            "",
            "the transcript of the video is:",
        ])
        if asr_transcript.strip():
            prompt_parts.append(asr_transcript)

        # Question section
        prompt_parts.extend([
            "",
            "question:",
            question,
            "",
            f"Please provide exactly {self.num_answers} different possible answers to this question.",
            f"Format your response as a numbered list from 1 to {self.num_answers}:",
            "",
        ])

        return "\n".join(prompt_parts)
    
    def create_target_answers(self, ground_truth: str) -> str:
        """Create target with ground truth and variations."""
        target_lines = [f"1. {ground_truth}"]
        
        # Simple repetition for now - in practice you'd want better variations
        for i in range(2, self.num_answers + 1):
            target_lines.append(f"{i}. {ground_truth}")
        
        return "\n".join(target_lines)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch for text-only training."""
        prompts = []
        targets = []
        ground_truths = []
        q_ids = []
        
        for feature in features:
            prompt = self.create_ag_prompt(
                question=feature["question"],
                asr_transcript=feature.get("asr_transcript", "")
            )
            target = self.create_target_answers(feature["ground_truth"])
            
            prompts.append(prompt)
            targets.append(target)
            ground_truths.append(feature["ground_truth"])
            q_ids.append(feature["Q_ID"])
        
        # Tokenize prompts and targets
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        target_encodings = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        
        # Concatenate and create labels
        input_len = prompt_encodings["input_ids"].size(1)
        target_len = target_encodings["input_ids"].size(1)
        
        full_input_ids = torch.cat([
            prompt_encodings["input_ids"],
            target_encodings["input_ids"]
        ], dim=1)
        
        full_attention = torch.cat([
            prompt_encodings["attention_mask"],
            target_encodings["attention_mask"]
        ], dim=1)
        
        # Create labels with input masked
        labels = torch.full_like(full_input_ids, -100)
        labels[:, input_len:input_len + target_len] = target_encodings["input_ids"]
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention,
            "labels": labels,
            "ground_truths": ground_truths,
            "q_ids": q_ids
        }

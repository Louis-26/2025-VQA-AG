"""
VQA model implementation for LoRA fine-tuned Qwen2.5-VL models.
Supports both transformers and vLLM backends with LoRA adapter loading.
"""

import os
import tempfile
import cv2
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import time

from .vqa_model_vllm import AnswerCandidate, BaseVQAModel


class LoRAQwenVQAModel(BaseVQAModel):
    """
    VQA model implementation for LoRA fine-tuned Qwen2.5-VL.
    Supports the trained 10-answer generation format.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.lora_adapter_path = model_config.get("lora_adapter_path")
        self.base_model_name = model_config.get("base_model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.use_vllm = model_config.get("engine") == "vllm"
        super().__init__(model_config)
    
    def _initialize_model(self):
        """Initialize the LoRA fine-tuned model."""
        if self.use_vllm:
            self._initialize_vllm_model()
        else:
            self._initialize_transformers_model()
    
    def _initialize_vllm_model(self):
        """Initialize vLLM with LoRA adapter support."""
        try:
            from vllm import LLM
            from transformers import AutoProcessor, AutoTokenizer
            
            # Check if LoRA adapter path exists
            if not self.lora_adapter_path or not os.path.exists(self.lora_adapter_path):
                raise ValueError(f"LoRA adapter path not found: {self.lora_adapter_path}")
            
            # Initialize vLLM with LoRA support
            # Note: vLLM v0.4+ supports LoRA adapters
            llm_kwargs = {
                "model": self.base_model_name,
                "trust_remote_code": True,
                "tensor_parallel_size": 1,  # Adjust based on your setup
                "enable_lora": True,
                "lora_modules": [
                    {
                        "name": "ag_lora",
                        "path": self.lora_adapter_path
                    }
                ]
            }
            
            self.model = LLM(**llm_kwargs)
            self.processor = AutoProcessor.from_pretrained(
                self.base_model_name, 
                trust_remote_code=True
            )
            
            print(f"Initialized vLLM with LoRA adapter: {self.lora_adapter_path}")
            
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        except Exception as e:
            print(f"Warning: vLLM LoRA initialization failed: {e}")
            print("Falling back to transformers implementation...")
            self._initialize_transformers_model()
    
    def _initialize_transformers_model(self):
        """Initialize transformers model with LoRA adapter."""
        from transformers import AutoTokenizer, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_name, 
            trust_remote_code=True
        )
        
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapter
        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            self.model = PeftModel.from_pretrained(base_model, self.lora_adapter_path)
            print(f"Loaded LoRA adapter from: {self.lora_adapter_path}")
        else:
            print("Warning: No LoRA adapter found, using base model")
            self.model = base_model
        
        self.model.eval()
        self.use_vllm = False  # Force transformers mode
    
    def _create_ag_prompt(self, question: str, asr_transcript: str = "") -> str:
        """
        Create the same prompt format used during training.
        This ensures the model generates the expected 10-answer format.
        """
        prompt_parts = [
            f"Question: {question}"
        ]
        
        if asr_transcript.strip():
            prompt_parts.append(f"ASR Transcript: {asr_transcript}")
        
        prompt_parts.extend([
            "",
            "Please provide exactly 10 different possible answers to this question based on the video content.",
            "Format your response as a numbered list from 1 to 10:",
            ""
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_numbered_answers(self, generated_text: str, num_answers: int = 10) -> List[str]:
        """
        Parse numbered answers from generated text.
        Same logic as used in training loss function.
        """
        pattern = r'^\d+\.?\s*(.+)$'
        lines = generated_text.strip().split('\n')
        
        answers = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try numbered format
            match = re.match(pattern, line)
            if match:
                answers.append(match.group(1).strip())
            else:
                # Fallback: treat each non-empty line as an answer
                if len(answers) < num_answers:
                    answers.append(line)
            
            if len(answers) >= num_answers:
                break
        
        # Ensure we have exactly num_answers
        while len(answers) < num_answers:
            if answers:
                answers.append(answers[-1])  # Repeat last answer
            else:
                answers.append("I cannot determine the answer from the video.")
        
        return answers[:num_answers]
    
    def generate_answers(
        self, 
        question: str, 
        frames: Optional[np.ndarray] = None, 
        video_path: Optional[str] = None,
        asr_transcript: str = "",
        num_answers: int = 10
    ) -> List[AnswerCandidate]:
        """
        Generate 10 answers using the fine-tuned model.
        """
        if not video_path:
            raise ValueError("LoRAQwenVQAModel requires a 'video_path'.")
        
        start_time = time.time()
        
        if self.use_vllm:
            return self._generate_with_vllm(question, video_path, asr_transcript, num_answers)
        else:
            return self._generate_with_transformers(question, video_path, asr_transcript, num_answers)
    
    def _generate_with_transformers(
        self, 
        question: str, 
        video_path: str,
        asr_transcript: str,
        num_answers: int
    ) -> List[AnswerCandidate]:
        """Generate answers using transformers backend."""
        start_time = time.time()
        
        # Create the training-format prompt
        prompt_text = self._create_ag_prompt(question, asr_transcript)
        
        # Use processor's built-in video handling
        video_uri = f"file://{os.path.abspath(video_path)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_uri},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text, 
            videos=[video_uri], 
            return_tensors="pt", 
            padding=True
        )
        
        # Generate with parameters similar to training
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode generated text
        input_length = inputs["input_ids"].size(1)
        generated_only = generated_ids[0][input_length:]
        generated_text = self.tokenizer.decode(generated_only, skip_special_tokens=True)
        
        # Parse answers
        answers = self._parse_numbered_answers(generated_text, num_answers)
        
        end_time = time.time()
        generation_time = (end_time - start_time) / len(answers)
        
        # Create AnswerCandidate objects
        candidates = []
        for i, answer in enumerate(answers):
            # Higher confidence for earlier answers (as trained)
            confidence = max(0.1, 1.0 - 0.05 * i)
            candidates.append(AnswerCandidate(
                text=answer.strip(),
                confidence=confidence,
                generation_time=generation_time
            ))
        
        return candidates
    
    def _generate_with_vllm(
        self, 
        question: str, 
        video_path: str,
        asr_transcript: str,
        num_answers: int
    ) -> List[AnswerCandidate]:
        """Generate answers using vLLM backend with LoRA."""
        from vllm import SamplingParams
        
        start_time = time.time()
        
        # Create the training-format prompt
        prompt_text = self._create_ag_prompt(question, asr_transcript)
        
        # Note: Current vLLM may not support video input directly
        # For now, we'll use text-only generation with the ASR transcript
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
            n=1,  # Generate one sequence with all 10 answers
        )
        
        # Generate with LoRA adapter
        outputs = self.model.generate(
            [text], 
            sampling_params,
            lora_request={
                "lora_name": "ag_lora",
                "lora_int_id": 1
            }
        )
        
        generated_text = outputs[0].outputs[0].text
        
        # Parse answers
        answers = self._parse_numbered_answers(generated_text, num_answers)
        
        end_time = time.time()
        generation_time = (end_time - start_time) / len(answers)
        
        # Create AnswerCandidate objects
        candidates = []
        for i, answer in enumerate(answers):
            confidence = max(0.1, 1.0 - 0.05 * i)
            candidates.append(AnswerCandidate(
                text=answer.strip(),
                confidence=confidence,
                generation_time=generation_time
            ))
        
        return candidates


def create_lora_vqa_model(lora_adapter_path: str, use_vllm: bool = False) -> LoRAQwenVQAModel:
    """
    Factory function to create a LoRA VQA model.
    
    Args:
        lora_adapter_path: Path to the LoRA adapter directory
        use_vllm: Whether to use vLLM backend (if available)
    
    Returns:
        Initialized LoRAQwenVQAModel
    """
    model_config = {
        "lora_adapter_path": lora_adapter_path,
        "base_model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "engine": "vllm" if use_vllm else "transformers",
        "max_length": 512,
        "description": f"LoRA fine-tuned Qwen2.5-VL from {lora_adapter_path}"
    }
    
    return LoRAQwenVQAModel(model_config)


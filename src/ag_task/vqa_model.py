from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class AnswerCandidate:
    """Represents a candidate answer with metadata for grounding and ranking."""
    text: str
    confidence: float
    generation_time: float
    grounding_evidence: Optional[Dict[str, Any]] = None  # For future grounding features
    
class BaseVQAModel(ABC):
    """Abstract base class for VQA models to enable easy model swapping."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model implementation."""
        pass
    
    @abstractmethod
    def generate_answers(self, frames: np.ndarray, question: str, 
                        num_answers: int = 10) -> List[AnswerCandidate]:
        """
        Generate ranked answer candidates for a video question.
        
        Args:
            frames: Video frames as numpy array (num_frames, height, width, 3)
            question: The question to answer
            num_answers: Number of diverse answers to generate
            
        Returns:
            List of AnswerCandidate objects sorted by confidence
        """
        pass
    
    def prepare_for_grounding(self, answer_candidates: List[AnswerCandidate], 
                            frames: np.ndarray) -> List[AnswerCandidate]:
        """
        Placeholder for future grounding implementation.
        Will add visual evidence retrieval and alignment checking.
        """
        # TODO: Implement retrieval-augmented grounding
        # TODO: Add cross-verification with video content
        return answer_candidates

class HuggingFaceVQAModel(BaseVQAModel):
    """Flexible HuggingFace-based VQA model implementation."""
    
    def _initialize_model(self):
        """Initialize HuggingFace model based on configuration."""
        from transformers import (
            VisionEncoderDecoderModel, 
            ViTImageProcessor, 
            AutoTokenizer,
            AutoModel,
            AutoProcessor
        )
        
        model_type = self.model_config.get("type", "vision_encoder_decoder")
        
        if model_type == "vision_encoder_decoder":
            vision_model = self.model_config.get("vision_model", "google/vit-base-patch16-224-in21k")
            text_model = self.model_config.get("text_model", "google/flan-t5-base")
            
            self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
                vision_model, text_model
            ).to(self.device)
            
            self.image_processor = ViTImageProcessor.from_pretrained(vision_model)
            self.tokenizer = AutoTokenizer.from_pretrained(text_model)
            
        elif model_type == "unified_multimodal":
            # For models like InstructBLIP, LLaVA, etc.
            model_name = self.model_config["model_name"]
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
        print(f"Initialized {model_type} model: {self.model_config}")
    
    def generate_answers(self, frames: np.ndarray, question: str, 
                        num_answers: int = 10) -> List[AnswerCandidate]:
        """Generate answer candidates using HuggingFace models."""
        import time
        
        if frames.size == 0:
            return [AnswerCandidate("Could not process video frames.", 0.0, 0.0)]
        
        start_time = time.time()
        
        if hasattr(self, 'processor'):  # Unified multimodal model
            answers = self._generate_with_unified_model(frames, question, num_answers)
        else:  # Vision-encoder-decoder model
            answers = self._generate_with_encoder_decoder(frames, question, num_answers)
        
        end_time = time.time()
        generation_time = (end_time - start_time) / len(answers)
        
        # Create AnswerCandidate objects with placeholder confidence scores
        candidates = []
        for i, answer in enumerate(answers):
            confidence = 1.0 - (i * 0.1)  # Simple confidence ranking
            candidates.append(AnswerCandidate(
                text=answer.strip(),
                confidence=max(confidence, 0.1),
                generation_time=generation_time
            ))
        
        return candidates
    
    def _generate_with_encoder_decoder(self, frames: np.ndarray, question: str, 
                                     num_answers: int) -> List[str]:
        """Generate answers using vision-encoder-decoder architecture."""
        # Convert frames to PIL Images for processing
        from PIL import Image
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Process frames
        pixel_values = self.image_processor(
            images=pil_frames, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Create prompt for answer generation
        prompt = f"Question: {question} Answer:"
        decoder_input_ids = self.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Generate diverse answers
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model_config.get("max_length", 50),
                num_beams=num_answers,
                num_return_sequences=num_answers,
                early_stopping=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    def _generate_with_unified_model(self, frames: np.ndarray, question: str, 
                                   num_answers: int) -> List[str]:
        """Generate answers using unified multimodal models (LLaVA, InstructBLIP, etc.)."""
        # TODO: Implement for specific unified models
        # This is a placeholder for future implementation
        return [f"Answer {i+1} for: {question[:30]}..." for i in range(num_answers)]

# Model factory for easy instantiation
def create_vqa_model(model_config: Dict[str, Any]) -> BaseVQAModel:
    """Factory function to create VQA models based on configuration."""
    model_family = model_config.get("family", "huggingface")
    
    if model_family == "huggingface":
        return HuggingFaceVQAModel(model_config)
    else:
        raise ValueError(f"Unsupported model family: {model_family}") 
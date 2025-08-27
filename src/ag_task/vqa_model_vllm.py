import os
import tempfile
import cv2
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import time
import multiprocessing

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
        self.device = "auto"
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model implementation."""
        pass
    
    @abstractmethod
    def generate_answers(self, question: str, frames: Optional[np.ndarray] = None, video_path: Optional[str] = None, 
                       num_answers: int = 10) -> List[AnswerCandidate]:
        """
        Generate ranked answer candidates for a video question.
        
        Args:
            question: The question to answer
            frames: Video frames as numpy array (num_frames, height, width, 3)
            video_path: Path to the video file
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

class QwenVQAModel(BaseVQAModel):
    """VQA model implementation specifically for Qwen-VL's native video processing."""
    
    def _initialize_model(self):
        """Initialize the Qwen-VL model."""
        from transformers import AutoTokenizer, AutoProcessor
        # Add reranker tokens so pointer IDs are single tokens
        try:
            from src.reranker.tokens import add_special_tokens_to_tokenizer
        except Exception:
            add_special_tokens_to_tokenizer = None
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        
        model_name = self.model_config["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).eval()
        if add_special_tokens_to_tokenizer is not None:
            try:
                add_special_tokens_to_tokenizer(self.tokenizer, self.model)
            except Exception:
                pass
        print(f"Initialized Qwen-VL model: {model_name}")

    def _normalize_answer(self, s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def generate_answers(self, question: str, frames: Optional[np.ndarray] = None, video_path: Optional[str] = None, 
                       num_answers: int = 1) -> List[AnswerCandidate]:
        """Generate answers using Qwen-VL's native video handling (supports multiple outputs)."""
        import time

        if not video_path:
            raise ValueError("QwenVQAModel requires a 'video_path'.")

        start_time = time.time()

        # Use processor's built-in video handling by passing a local file URI.
        video_uri = f"file://{os.path.abspath(video_path)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_uri},
                    {"type": "text", "text": f"Answer the following question concisely in one sentence: {question}"}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Let the processor decode the video path. Avoid manual OpenCV decoding.
        inputs = self.processor(text=text, videos=[video_uri], return_tensors="pt", padding=True)
        # Don't force .to("auto"); rely on HF moving tensors during generation.
        
        # Generate multiple sequences with sampling
        gen_kwargs = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=64,
            # num_return_sequences=max(1, int(num_answers)),
            num_return_sequences=1,

        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim input prompt tokens before decoding
        generated_ids_trimmed = [
            out_ids[len(inputs.input_ids[0]):] for out_ids in generated_ids
        ]
        decoded = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # De-duplicate to <=10
        unique_texts: List[str] = []
        seen = set()
        for ans in decoded:
            norm = self._normalize_answer(ans)
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            unique_texts.append(ans.strip())
            if len(unique_texts) >= 10:
                break
        if not unique_texts:
            unique_texts = [decoded[0].strip()] if decoded else [""]
        
        end_time = time.time()
        per_ans_time = (end_time - start_time) / max(1, len(unique_texts))
        
        candidates: List[AnswerCandidate] = []
        # Simple descending placeholder confidence
        for i, ans in enumerate(unique_texts):
            conf = max(0.1, 1.0 - 0.05 * i)
            candidates.append(AnswerCandidate(text=ans, confidence=conf, generation_time=per_ans_time))
        return candidates


class HuggingFaceVQAModel(BaseVQAModel):
    """Flexible HuggingFace-based VQA model implementation."""
    
    def _initialize_model(self):
        """Initialize HuggingFace model based on configuration."""
        from transformers import (
            VisionEncoderDecoderModel, 
            ViTImageProcessor, 
            AutoTokenizer,
            AutoModelForCausalLM
        )
        try:
            from src.reranker.tokens import add_special_tokens_to_tokenizer
        except Exception:
            add_special_tokens_to_tokenizer = None
        
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
            from transformers import AutoProcessor
            # For models like InstructBLIP, LLaVA, etc.
            model_name = self.model_config["model_name"]
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if add_special_tokens_to_tokenizer is not None:
            try:
                add_special_tokens_to_tokenizer(self.tokenizer, self.model)
            except Exception:
                pass
        
        print(f"Initialized {model_type} model: {self.model_config['model_name']}")

    def generate_answers(self, frames: np.ndarray, question: str, 
                        num_answers: int = 10) -> List[AnswerCandidate]:
        """Generate answer candidates using HuggingFace models."""
        import time
        
        if frames.size == 0:
            return [AnswerCandidate("Could not process video frames.", 0.0, 0.0)]
        
        start_time = time.time()
        
        model_type = self.model_config.get("type")

        if model_type == "unified_multimodal":
            answers = self._generate_with_unified_model(frames, question, num_answers)
        elif model_type == "vision_encoder_decoder":
            answers = self._generate_with_encoder_decoder(frames, question, num_answers)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
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

    def _save_frames_to_temp_files(self, frames: np.ndarray, num_keyframes: int) -> List[str]:
        """Saves a selection of frames to temporary files and returns their paths."""
        temp_files = []
        if len(frames) == 0:
            return []

        # Select evenly spaced keyframes
        indices = np.linspace(0, len(frames) - 1, num=num_keyframes, dtype=int)
        
        for i, frame_idx in enumerate(indices):
            frame = frames[frame_idx]
            # Use a context manager that handles cleanup
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp_file.name, frame)
            temp_files.append(temp_file.name)
        return temp_files

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Removes the temporary image files."""
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error removing temp file {path}: {e}")
    
    def _generate_with_unified_model(self, frames: np.ndarray, question: str, 
                                   num_answers: int) -> List[str]:
        """Generate answers using unified multimodal models (LLaVA, InstructBLIP, etc.)."""
        num_keyframes = 3  # Use 3 frames: start, middle, end
        temp_image_paths = self._save_frames_to_temp_files(frames, num_keyframes)
        
        if not temp_image_paths:
            return ["Could not process video frames."]

        try:
            # 2. Prepare the prompt for the model using the file paths
            prompt_list = []
            for i, path in enumerate(temp_image_paths):
                prompt_list.append({'image': path})
            
            prompt_list.append({'text': f"Based on this sequence of images, answer the following question: {question}"})
            
            query = self.tokenizer.from_list_format(prompt_list)

            # 3. Generate a response
            with torch.no_grad():
                inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=100)
                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Clean up the response by removing the echoed prompt
                # The model often includes the original question in its response
                clean_response = response_text.split(question)[-1].strip()
                if not clean_response:  # If split fails, use original response
                    clean_response = response_text
                
        finally:
            # 4. Clean up the temporary files
            self._cleanup_temp_files(temp_image_paths)
        
        return [clean_response]

class VLLMVQAModel(BaseVQAModel):
    """VQA model implementation specifically for vLLM."""
    
    def _initialize_model(self):
        """Initialize the vLLM model."""
        from vllm import LLM
        from transformers import AutoProcessor, AutoTokenizer
        from tempfile import mkdtemp
        try:
            from src.reranker.tokens import add_special_tokens_to_tokenizer
        except Exception:
            add_special_tokens_to_tokenizer = None
        
        model_name = self.model_config["model_name"]
        
        tokenizer_path = None
        if add_special_tokens_to_tokenizer is not None:
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                add_special_tokens_to_tokenizer(tok, None)
                tokenizer_path = mkdtemp(prefix="qwen_tok_")
                tok.save_pretrained(tokenizer_path)
            except Exception:
                tokenizer_path = None
        # vLLM engine initialization (pass tokenizer path if available so special tokens are recognized)
        llm_kwargs = dict(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=4
        )
        if tokenizer_path:
            llm_kwargs["tokenizer"] = tokenizer_path
        self.model = LLM(**llm_kwargs)
        
        # The processor is still needed to format the prompt correctly
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Initialized vLLM engine for model: {model_name}")
    
    def generate_answers(self, question: str, frames: Optional[np.ndarray] = None, video_path: Optional[str] = None, 
                       num_answers: int = 10) -> List[AnswerCandidate]:
        """Generate answers using vLLM.
        Note: vLLM path currently treats input as text-only; video content is not ingested here.
        """
        from vllm import SamplingParams
        
        if not video_path:
            raise ValueError("VLLMVQAModel requires a 'video_path'.")
        
        start_time = time.time()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Answer the following question concisely in one sentence: {question}"}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # inputs = self.processor(text=text, videos=[video_data], return_tensors="pt", padding=True)

        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=self.model_config.get("temperature", 0.7),
            max_tokens=self.model_config.get("max_new_tokens", 64),
            top_p=0.9,
            n=num_answers,
        )
        
        # Generate answers
        with torch.no_grad():
            outputs = self.model.generate([text], sampling_params)
        end_time = time.time()
        generation_time = (end_time - start_time) / len(outputs)
        # Process the outputs
        answers = []
        for output in outputs[0].outputs:
            text = output.text
            answers.append(AnswerCandidate(text=text, confidence=1.0, generation_time=generation_time))
        
        return answers
    
    def _extract_frames_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """Extract frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames_list: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        return np.array(frames_list)


# Model factory for easy instantiation
def create_vqa_model(model_config: Dict[str, Any]) -> BaseVQAModel:
    """Factory function to create VQA models based on configuration."""
    model_name = model_config.get("model_name", "")
    model_family = model_config.get("family", "huggingface")
    engine = model_config.get("engine", "")
    
    # Handle LoRA fine-tuned models
    if model_family == "lora":
        from .vqa_model_lora import LoRAQwenVQAModel
        return LoRAQwenVQAModel(model_config)
    
    # Handle vLLM engine
    if engine == "vllm":
        return VLLMVQAModel(model_config)
    
    # Handle Qwen models
    if "qwen" in model_name.lower() and "vl" in model_name.lower():
        return QwenVQAModel(model_config)
    
    # Handle standard HuggingFace models
    if model_family == "huggingface":
        return HuggingFaceVQAModel(model_config)
    else:
        raise ValueError(f"Unsupported model family: {model_family}") 


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # The context might already be set in some environments (e.g., Jupyter).
        pass
    vllm_config = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "engine": "vllm",          
        "num_keyframes": 5,
        "temperature": 0.7,
        "max_new_tokens": 64,
    }
    model = create_vqa_model(vllm_config)
    question = "What is the video about?"

    answers = model.generate_answers(question, video_path="./test.mp4", num_answers=10)
    print(answers)
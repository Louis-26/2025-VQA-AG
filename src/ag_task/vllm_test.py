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

# Suppress a specific warning from transformers if it occurs
import warnings
warnings.filterwarnings("ignore", message="The 'pretraining_tp' field.*")

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
        # vLLM manages its own device placement, so this is less critical
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "auto"
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific model implementation."""
        pass
    
    @abstractmethod
    def generate_answers(self, question: str, frames: Optional[np.ndarray] = None, video_path: Optional[str] = None, 
                       num_answers: int = 1) -> List[AnswerCandidate]:
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
        return answer_candidates

    def _extract_frames_from_video(self, video_path: str) -> Optional[np.ndarray]:
        """Utility to extract frames from a video path."""
        if not os.path.exists(video_path):
            print(f"Video path does not exist: {video_path}")
            return None
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return None
        
        frames_list: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert from BGR (cv2 default) to RGB
            frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames_list:
            print(f"No frames extracted from video: {video_path}")
            return None
            
        return np.array(frames_list)

class VLLMVQAModel(BaseVQAModel):
    """
    VQA implementation using the vLLM engine for high-performance inference.
    This model processes video by sampling keyframes and passing their file paths
    to a multi-modal LLM like Qwen-VL or LLaVA.
    """
    def _initialize_model(self):
        """Initializes the vLLM engine and the corresponding HuggingFace processor."""
        from vllm import LLM
        from transformers import AutoProcessor
        
        model_name = self.model_config["model_name"]
        
        # vLLM engine initialization
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            # For multi-GPU, you can set tensor_parallel_size
            tensor_parallel_size=4
        )
        
        # The processor is still needed to format the prompt correctly
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Initialized vLLM engine for model: {model_name}")

    def generate_answers(self, question: str, frames: Optional[np.ndarray] = None, video_path: Optional[str] = None,
                       num_answers: int = 1) -> List[AnswerCandidate]:
        """Generate answers using the vLLM engine with frame sampling."""
        from vllm import SamplingParams

        if frames is None and video_path:
            frames = self._extract_frames_from_video(video_path)

        if frames is None or frames.size == 0:
            return [AnswerCandidate("Could not process video frames.", 0.0, 0.0)]
        
        start_time = time.time()
        
        # 1. Save keyframes to temporary files
        num_keyframes = self.model_config.get("num_keyframes", 3)
        temp_image_paths = self._save_frames_to_temp_files(frames, num_keyframes)
        
        if not temp_image_paths:
            return [AnswerCandidate("Failed to save temporary frames.", 0.0, 0.0)]

        answers = []
        try:
            # 2. Prepare the prompt for the model
            # For multi-modal models, this usually involves special tokens like <image>
            # The AutoProcessor handles this formatting.
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
            video_data = np.array(frames_list)
            messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_data},
                    {"type": "text", "text": f"Answer the following question concisely in one sentence: {question}"}
                ]
            }
            ]
            # The processor creates the final string with image placeholders
            final_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 3. Define sampling parameters for vLLM
            sampling_params = SamplingParams(
                n=num_answers,
                temperature=self.model_config.get("temperature", 0.7),
                top_p=self.model_config.get("top_p", 0.9),
                max_tokens=self.model_config.get("max_new_tokens", 100)
            )

            # 4. Generate a response using the vLLM engine
            outputs = self.model.generate([final_prompt], sampling_params)

            # Extract text from all generated sequences
            for output in outputs[0].outputs:
                answers.append(output.text)

        finally:
            # 5. Clean up the temporary files
            self._cleanup_temp_files(temp_image_paths)
            
        end_time = time.time()
        generation_time = (end_time - start_time) / max(1, len(answers))
        
        candidates = []
        for i, answer_text in enumerate(answers):
            confidence = 1.0 - (i * 0.1) # Simple placeholder confidence
            candidates.append(AnswerCandidate(
                text=answer_text.strip(),
                confidence=max(confidence, 0.1),
                generation_time=generation_time
            ))
            
        return candidates

    def _save_frames_to_temp_files(self, frames: np.ndarray, num_keyframes: int) -> List[str]:
        """Saves a selection of frames to temporary files and returns their paths."""
        temp_files = []
        if len(frames) == 0:
            return []

        indices = np.linspace(0, len(frames) - 1, num=num_keyframes, dtype=int)
        
        for frame_idx in indices:
            frame = frames[frame_idx]
            # Convert back to BGR for cv2.imwrite
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, frame_bgr)
                temp_files.append(temp_file.name)
        return temp_files

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Removes the temporary image files."""
        for path in file_paths:
            try:
                os.remove(path)
            except OSError as e:
                print(f"Error removing temp file {path}: {e}")


# Model factory for easy instantiation
def create_vqa_model(model_config: Dict[str, Any]) -> BaseVQAModel:
    """Factory function to create VQA models based on configuration."""
    engine = model_config.get("engine")
    
    if engine == "vllm":
        # Ensure a compatible model is being used
        model_name = model_config.get("model_name", "").lower()
        if "qwen" in model_name or "llava" in model_name:
             return VLLMVQAModel(model_config)
        else:
            raise ValueError(f"Model {model_config.get('model_name')} may not be compatible with the vLLM image-processing logic.")
    
    # Fallback to original logic if engine is not specified or different
    model_name = model_config.get("model_name", "")
    if "qwen" in model_name.lower() and "vl" in model_name.lower():
        # Note: The original QwenVQAModel is not included here for brevity,
        # but would handle native video processing if kept.
        # For this example, we assume vLLM is the primary way to use Qwen.
        raise NotImplementedError("Original QwenVQAModel not included in this script. Use engine='vllm'.")
    
    model_family = model_config.get("family", "huggingface")
    if model_family == "huggingface":
        # The original HuggingFaceVQAModel is also not included,
        # as the vLLM class now provides a superior implementation path.
        raise NotImplementedError("Original HuggingFaceVQAModel not included. Use engine='vllm'.")
    else:
        raise ValueError(f"Unsupported model family or engine: {model_family} / {engine}")

# --- Example Usage ---
def create_dummy_video(path="dummy_video.mp4", frames=30):
    """Creates a simple dummy video for testing."""
    width, height = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    for i in range(frames):
        # Create a frame with a moving red square
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x_pos = int((i / frames) * width)
        cv2.rectangle(frame, (x_pos, 100), (x_pos + 20, 120), (0, 0, 255), -1) # BGR format for cv2
        out.write(frame)
    out.release()
    print(f"Dummy video created at {path}")


if __name__ == '__main__':
    # Make sure you have installed vllm: pip install vllm
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # The context might already be set in some environments (e.g., Jupyter).
        pass
    if not True:
        print("vLLM requires a CUDA-enabled GPU. Skipping example.")
    else:
        # Create a dummy video for the example
        video_file = "/brtx/603-nvme1/yweng13/VQA/my_train_videos/J7C_J_BOq7I.mp4"
        create_dummy_video(video_file, frames=50)

        # 1. Define the model configuration for a vLLM-compatible multimodal model
        # Using Qwen2-VL as an example.
        vllm_config = {
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "engine": "vllm",          # <-- Key to select the vLLM engine
            "num_keyframes": 3,        # Sample 3 frames from the video
            "temperature": 0.2,
            "max_new_tokens": 50,
        }

        try:
            # 2. Create the VQA model using the factory
            print("\nInitializing VQA model with vLLM engine...")
            vqa_model = create_vqa_model(vllm_config)

            # 3. Ask a question about the video
            question = "summarize the video"
            print(f"\nQuestion: {question}")
            
            # Generate answers
            # The model can take either a path or pre-loaded frames
            answer_candidates = vqa_model.generate_answers(
                question=question,
                video_path=video_file,
                num_answers=10 # Ask for 10 different answers
            )

            # 4. Print the results
            print("\nGenerated Answers:")
            if answer_candidates:
                for i, candidate in enumerate(answer_candidates):
                    print(f"  Answer {i+1}: '{candidate.text}'")
                    print(f"    - Confidence (placeholder): {candidate.confidence:.2f}")
                    print(f"    - Generation Time (avg): {candidate.generation_time:.4f}s")
            else:
                print("No answers were generated.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure 'vllm' is installed (`pip install vllm`) and you have a compatible GPU setup.")
        finally:
            # Clean up the dummy video file
            pass
"""
Model configurations for different VQA approaches.
This allows easy experimentation with different models and settings.
"""

from typing import Dict, Any

# Baseline configurations for different model types
MODEL_CONFIGS = {
    "baseline_encoder_decoder": {
        "family": "huggingface",
        "type": "vision_encoder_decoder",
        "vision_model": "google/vit-base-patch16-224-in21k",
        "text_model": "google/flan-t5-base",
        "max_length": 50,
        "description": "Basic ViT + T5 encoder-decoder model"
    },
    
    "improved_encoder_decoder": {
        "family": "huggingface", 
        "type": "vision_encoder_decoder",
        "vision_model": "google/vit-large-patch16-224",
        "text_model": "google/flan-t5-large",
        "max_length": 100,
        "description": "Larger ViT + T5 for better performance"
    },
    
    # Placeholder for future unified multimodal models
    "instructblip": {
        "family": "huggingface",
        "type": "unified_multimodal", 
        "model_name": "Salesforce/instructblip-vicuna-7b",
        "max_length": 100,
        "description": "InstructBLIP for video QA"
    },
    
    "llava": {
        "family": "huggingface",
        "type": "unified_multimodal",
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "max_length": 100,
        "description": "LLaVA multimodal model"
    },
    
    "qwen_vl_chat": {
        "family": "huggingface",
        "type": "unified_multimodal",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_length": 100,
        "description": "Official Qwen 2.5 VL 7B Instruct model"
    },
    "qwen_vl_chat_vllm": {
        "family": "huggingface",
        "type": "unified_multimodal",
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "max_length": 100,
        "engine": "vllm",
        "description": "Official Qwen 2.5 VL 7B Instruct model with vLLM engine"
    },
    
    # Configuration for future grounding-enhanced models
    "grounded_vqa": {
        "family": "huggingface",
        "type": "vision_encoder_decoder",
        "vision_model": "google/vit-base-patch16-224-in21k", 
        "text_model": "google/flan-t5-base",
        "max_length": 75,
        "enable_grounding": True,
        "grounding_model": "clip-vit-base-patch32",  # For future frame retrieval
        "description": "VQA with grounding and evidence retrieval"
    }
}

def get_model_config(config_name: str) -> Dict[str, Any]:
    """Get a model configuration by name."""
    if config_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return MODEL_CONFIGS[config_name].copy()

def list_available_configs() -> Dict[str, str]:
    """List all available model configurations with descriptions."""
    return {name: config["description"] for name, config in MODEL_CONFIGS.items()}
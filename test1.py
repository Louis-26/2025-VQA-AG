from llava.model.builder import load_pretrained_model
from transformers.modeling_utils import PreTrainedModel

# Monkeypatch to handle dict-based text_config during embedding resize
_orig_resize = PreTrainedModel.resize_token_embeddings

def _safe_resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, *args, **kwargs):
    text_cfg = getattr(self.config, "text_config", None)
    if isinstance(text_cfg, dict):
        class _CfgObj:
            pass
        obj = _CfgObj()
        for k, v in text_cfg.items():
            setattr(obj, k, v)
        self.config.text_config = obj
    return _orig_resize(self, new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of, *args, **kwargs)

PreTrainedModel.resize_token_embeddings = _safe_resize_token_embeddings

tok, model, img_proc, max_len = load_pretrained_model(
    model_path="lmms-lab/llava-critic-7b",
    model_base=None,
    model_name="llava_qwen",
    device_map="auto",
    attn_implementation="sdpa",
)
print("Loaded:", type(model).__name__, "max_len:", max_len)
from llava.model.builder import load_pretrained_model

tok, model, img_proc, max_len = load_pretrained_model(
    model_path="lmms-lab/llava-critic-7b",
    model_base=None,
    model_name="llava_qwen",
    device_map="auto",
    attn_implementation="sdpa",
)
print("Loaded:", type(model).__name__, "max_len:", max_len)
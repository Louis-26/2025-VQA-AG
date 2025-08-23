# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch, soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os
import pathlib
from pathlib import Path
import tempfile
import yt_dlp
import subprocess
import json
import gc, torch
import sys


def free_cuda(tag=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        print(f"[{tag}] free {free/1024**3:.2f} GB / total {total/1024**3:.2f} GB | "
              f"alloc {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
              f"reserved {torch.cuda.memory_reserved()/1024**3:.2f} GB")

VIDEO_DIR = "/brtx/603-nvme1/yweng13/VQA/my_train_videos"
JSON_DIR = "/brtx/603-nvme1/yweng13/VQA/train_json_files"

# %%
# Load model
free_cuda("start")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2" 
).eval()

model.gradient_checkpointing_enable()
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-Omni-3B",
#     torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
#     device_map="auto",
#     max_memory={0: "18GiB", "cpu": "32GiB"},  # Leave some GPU memory free
#     offload_folder="./model_offload",  # This handles offloading properly
#     offload_buffers=True,  # Offload buffers too
#     low_cpu_mem_usage=True,
# ).eval()


# %%
prompt = """

You are a video question answering model. You will be given a video URL and a question about the video. Your task has 2 steps:

(1) Identify the correct answer to the provided question based on the content of the video. If it cannot be answered, you are to indicate that the question is not answerable by returning "Cannot be answered" as the answer. Your answer should be extremely syntactically similar to how the examples below are written.
(2) Paraphrase the correct answer to the question 9 different times to create 9 incorrect answers. These answers should be semantically simialar to the correct answer, possibly includeing the correct answer inside them, but with extra information taht makes them incorrect. For example, if the correct answer is "Washing your hair twice gives you cleaner hair and better lather," and incorrect answer could be "Washing hair  twice puts bugs in your hair."

You will output your answer in a json in the following format:

{
  "correct_answer": <the correct answer to the question>,
  "incorrect_answers": [ <9 incorrect answers to the question> ]
}

Here is an example of a properly filled out response. All of your answers should try and match the writing style of the examples below as closely as possible. Look at how they phrase their answers at match that semantic style when generating yours. For brevity, they only have 5 total answers. Yours will need to have 10.

{
  "correct_answer": "The father begins to choke of them regretting he had asked for some.",
  "incorrect_answers": [
    "The man threw up.",
    "The man ate the whole bowl.",
    "The man threw it in the trash."
  ],
}

{
  "correct_answer": "After kissing in a car they are kissing in the ocean.",
  "incorrect_answers": [
    "They go to dinner.",
    "The sit on a beach.",
    "They look for their dog."
  ],
}

{
  "question": "Why do the men keep riding their motorcycles through water?",
  "correct_answer": "The motorcycles are made for water.",
  "incorrect_answers": [
    "They are better on dry land.",
    "They are test driving the motorcycles.",
    "They can climb up trees."
  ],
}

{
  "correct_answer": "The two men shook hands and bumped each other in a form of agreement and brotherhood.",
  "incorrect_answers": [
    "The two men shook hands and bumped each other because The two men shook hands and bumped each other because they were upset.\n",
    "The two men shook hands and bumped each other because the were crying.",
    "The two men shook hands and bumped each other because  their wives joined them."
  ],
}

{
  "video_url": "https://www.youtube.com/watch?v=CbRNu0FBRv8",
  "question": "when does the man scroll the screen  on his phone?",
  "correct_answer": "Before showing the screen to the woman",
  "incorrect_answers": [
    "After showing the screen to the woman",
    "As he shows the screen to the woman",
    "he does not scroll at all"
  ],
}

"""

# %%
def find_mp4_for_json(file_stem, video_dir):
        matches = sorted(Path(video_dir).glob(f"{file_stem}*.mp4"))
        if not matches:
            return None
        return matches[0]

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    return json.loads(text.split("assistant\n")[-1].strip())

def generate_qa(file_name, prompt, question):
    work = pathlib.Path(tempfile.mkdtemp())
    processed_video_path = work / f"{file_name}_processed.mp4"

    print(file_name)

    video_path = find_mp4_for_json(file_name, VIDEO_DIR)

    # Downsample video until something works
    fps_candidates = [15, 12, 10, 8, 4, 2, 1]
    #height_candidates = [720, 480, 360, 240]

    for fps in fps_candidates:
        #for h in height_candidates:
                try:
                    cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(video_path), #input 0
                        #"-i", str(audio_path), #input 1
                        "-vf", f"fps={fps}", #"scale=-2:{h}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-movflags", "+faststart",
                        #"-map", "0:v:0", #take video from input 0
                        #"-map", "1:a:0", #take audio from input 1
                        #"-c:a", "aac", "-b:a", "96k", "-ac", str(ac),
                        #"-shortest", #stop when the shorter stream ends
                        str(processed_video_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)

                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": question}]},
                        {"role": "user", "content": [{"type": "video", "video": str(processed_video_path)}]},
                    ]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

                    inputs_cpu = processor(
                        text=text,
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=False,
                        use_audio_in_video=False,
                    )

                    inputs_gpu = {}
                    for key, value in inputs_cpu.items():
                        if isinstance(value, torch.Tensor):
                            if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                inputs_gpu[key] = value.to(model.device).to(model.dtype)
                            else:
                                inputs_gpu[key] = value.to(model.device)
                        else:
                            inputs_gpu[key] = value

                    free_cuda("pre-generate")

                    with torch.no_grad():
                        text_ids = model.generate(
                            **inputs_gpu,
                            cache_implementation="offloaded",
                            use_audio_in_video=False,
                            return_audio=False
                        )

                    answer = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    print(answer)
                    return answer #yay!

                except torch.cuda.OutOfMemoryError as e:
                    free_cuda(f"failed fps = {fps}")
                    continue  #try next combo

    # Exhausted all combos
    raise RuntimeError("Ran out of GPU memory for all fps/ac combinations tried.")


# %%
from pathlib import Path
import json

def combined_data(original, answers):
    new = {}
    new['video_url'] = original['video_url']
    new['question'] = original['question']
    new['correct_answer'] = answers['correct_answer']
    new['incorrect_answers'] = answers['incorrect_answers']
    return new


for file in Path(JSON_DIR).iterdir():
    # Clear everything before each video
    free_cuda("loop-start")
    
    file_name = file.name
    print(file_name)
    
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        if file.suffix != ".json":
            print(f"Skipping non-JSON file: {file}")
            continue
        else:
            print(f"Failed to read JSON file {file}!!")
            break
    
    print(data["video_url"], data["question"])
    

    answers = generate_qa(file.stem, prompt, data["question"])

    print("\n\n\n\n")

    print("Generated answer:")
    print(answers)

    print("\n\n\n\n")

    
    # Parse and save
    parsed_data = extract_json_from_text(answers)
    final_json = combined_data(data, parsed_data)
    with open(f"/brtx/605-nvme1/kguerre6/quen2.5Omni_QA/{file_name}", "w") as f:
        json.dump(final_json, f, indent=2)
    
    # Cleanup after each video
    free_cuda("loop-end")

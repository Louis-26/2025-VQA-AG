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

# %%
# Load model
free_cuda("start")

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

model.gradient_checkpointing_enable()
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

You are a video question answering model. You will be given a video URL and a question about the video. Your task has 6 steps:

(1) Identify the correct answer to the provided question based on the content of the video. If it cannot be answered, you are to indicate that the question is not answerable by returning "Cannot be answered" as the answer.
(2) Ground your answer in a specific timestamp of the video that you found the answer in.
(3) Determine the quality of the answer based on how well it is supported by the video content. The quality can be "high", "medium", or "low". A high quality answer is one that is directly supported by the video content, a medium quality answer is one that is somewhat supported by the video content, and a low quality answer is one that is not well supported by the video content.
(4) Determine whether the question can be answered from the video content. If you were able to answer the question from step (1), set answerable to true. If your response was "Cannot be answered", set the answerable field to false
(5) Determine the modality that the answer was found. Your options are "video," "text," or "audio." You should include as many as possible, but if you can only find the answer in one modality, you should only include that one. If the answer was found in the video content, set the modality to "video". If the answer was found in the audio content, set the modality to "audio". If the answer was found in the text content, set the modality to "text". If you could not answer the question, set the modality to "none."
(6) Paraphrase the correct answer to the question 4 different times to create 4 incorrect answers. These answers should be semantically simialar to the correct answer, possibly includeing the correct answer inside them, but with extra information taht makes them incorrect. For example, if the correct answer is "Washing your hair twice gives you cleaner hair and better lather," and incorrect answer could be "Washing hair  twice puts bugs in your hair."

You will output your answer in a json in the following format:

{
  "video_url": <the url provided in the prompt>,
  "question": <the question you are answering, provided in the prompt>,
  "correct_answer": <the correct answer to the question>,
  "incorrect_answers": [ <4 incorrect answers to the question> ],
  "timestamp": <the timestamp in the video where the answer can be found>,
  "quality": <the quality of the answer, either "high", "medium", or "low">,
  "answerable": <true or false, whether the question can be answered from the video>,
  "modality": <"video", "audio", or "text", depending on the modality in which you found the answer>,
}

Here is an example of a properly filled out response:

{
  "video_url": "https://www.youtube.com/watch?v=BT1-7xs4k3Y",
  "question": "What happens after a father takes a bite of his daughters noodles?",
  "correct_answer": "The father begins to choke of them regretting he had asked for some.",
  "incorrect_answers": [
    "The man threw up.",
    "The man ate the whole bowl.",
    "The man threw it in the trash."
  ],
  "timestamp": "2025-07-14T00:57:02.017Z",
  "quality": "poor",
  "answerable": false,
  "modality": "audio or video"
}

"""

# %%
def download_video(video_url, output_path):
    print(f"Downloading video from {video_url}...")
    try:
        subprocess.run(
            [
                "yt-dlp",
                "-f", "bv*+ba/b",  #select best video and audio (don't have a best that has both)
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                video_url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during video download: {e.stderr}")


def generate_qa(file_name, prompt, video_url, question):
    work = pathlib.Path(tempfile.mkdtemp())
    original_video_path = work / f"{file_name}_original.mp4"
    processed_video_path = work / f"{file_name}_processed.mp4"

    print(original_video_path)

    # Download the video 
    download_video(video_url, original_video_path)

    # Downsample audio/video until something works
    fps_candidates = [15, 12, 10, 8, 4, 2, 1]
    ac_candidates = [8, 6, 4, 2, 1]
    height_candidates = [720, 480, 360, 240]
    last_oom = None
    last_ffmpeg_err = None

    for fps in fps_candidates:
        for h in height_candidates:
            for ac in ac_candidates:
                try:
                    cmd = [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(original_video_path),
                        "-vf", f"fps={fps},scale=-2:{h}",   # reduce FPS, then scale to target height
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-movflags", "+faststart",
                        "-map", "0:v:0", "-map", "0:a:0?",
                        "-c:a", "aac", "-b:a", "96k", "-ac", str(ac),
                        str(processed_video_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)

                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": prompt}]},
                        {"role": "user", "content": [{"type": "text", "text": question}]},
                        {"role": "user", "content": [{"type": "video", "video": str(processed_video_path)}]},
                    ]

                    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

                    inputs_cpu = processor(
                        text=text,
                        audio=audios,
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                        padding=False,
                        use_audio_in_video=True,
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
                            use_audio_in_video=True
                        )

                    answer = processor.batch_decode(
                        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                    print(answer)
                    return answer #yay!

                except torch.cuda.OutOfMemoryError as e:
                    last_oom = e
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue  #try next combo

                except subprocess.CalledProcessError as e:
                    # ffmpeg failed
                    last_ffmpeg_err = e
                    continue

                # Cleanup
                finally:
                    try:
                        if original_video_path.exists():
                            os.remove(original_video_path)
                    except Exception:
                        pass
                    try:
                        if processed_video_path.exists():
                            os.remove(processed_video_path)
                    except Exception:
                        pass

    # Exhausted all combos
    raise RuntimeError("Ran out of GPU memory for all fps/ac combinations tried.")


# %%
from pathlib import Path
import json

folder_path = Path("/brtx/605-nvme1/kguerre6/alex_checked")

for file in folder_path.iterdir():
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
    

    answers = generate_qa(file.stem, prompt, data["video_url"], data["question"])
    
    # Parse and save
    parsed_data = json.loads(answers)
    with open(f"/brtx/605-nvme1/kguerre6/quen2.5Omni_QA/{file_name}", "w") as f:
        json.dump(parsed_data, f, indent=2)
    
    # Cleanup after each video
    free_cuda("loop-end")
    
    break

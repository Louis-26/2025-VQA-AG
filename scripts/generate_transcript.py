import whisper
import sys
import os
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ag_task.json_data_loader import load_json_topics
import argparse
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser(description="Generate transcript for a video.")
    parser.add_argument("--json_files_dir", type=str, required=True, 
                       help="Path to the directory with JSON files.")

    parser.add_argument("--videos_dir", type=str, required=True, 
                       help="Path to the directory with videos.")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Path to the directory to save the transcripts.")

    args = parser.parse_args()


    topics = load_json_topics(args.json_files_dir)
    model = whisper.load_model("turbo")

    for topic in tqdm(topics):
        video_id = topic['Video_ID']
        video_path = os.path.join(args.videos_dir, video_id + '.mp4')
        if not os.path.exists(video_path):
            print(f"Video {video_id} not found")
            continue
        try:
            result = model.transcribe(video_path)
            transcript = result['text']
            topic['transcript'] = transcript
        except Exception as e:
            print(f"Error transcribing video {video_id}: {e}")
            continue

    with open(os.path.join(args.output_dir, 'transcripts.json'), 'w') as f:
        json.dump(topics, f)



if __name__ == "__main__":
    main()


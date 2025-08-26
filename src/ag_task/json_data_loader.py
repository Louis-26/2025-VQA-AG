import os
import json
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs

def load_json_topics(json_files_dir: str) -> List[Dict[str, Any]]:
    """
    Loads and parses the JSON topics from a directory of JSON files.

    Args:
        json_files_dir (str): The path to the directory containing JSON topic files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               represents a question with its metadata.
    """
    topics = []
    for filename in os.listdir(json_files_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_files_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                video_url = data.get("video_url")
                if not video_url:
                    print(f"Warning: 'video_url' not found in {filename}. Skipping.")
                    continue

                # Extract Video_ID from the YouTube URL
                parsed_url = urlparse(video_url)
                video_id = parse_qs(parsed_url.query).get('v')
                
                if not video_id:
                    print(f"Warning: Could not parse Video_ID from URL {video_url} in {filename}. Skipping.")
                    continue

                topics.append({
                    "Q_ID": filename.replace('.json', ''),
                    "Video_ID": video_id[0],
                    "Question": data["question"],
                    "correct_answer": data["correct_answer"],
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process file {filename}. Error: {e}. Skipping.")
                continue
                
    return topics







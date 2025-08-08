import cv2
import numpy as np
from typing import List

def extract_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """
    Extracts a specified number of frames evenly spaced from a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): The number of frames to extract.

    Returns:
        np.ndarray: A numpy array of shape (num_frames, height, width, 3)
                    containing the extracted frames. Returns an empty array if
                    the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return np.array([])

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    frame_idx_iterator = iter(frame_indices)
    target_frame_idx = next(frame_idx_iterator, None)
    
    current_frame_num = 0
    while(cap.isOpened() and target_frame_idx is not None):
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_frame_num == target_frame_idx:
            frames.append(frame)
            target_frame_idx = next(frame_idx_iterator, None)
        
        current_frame_num += 1
    
    cap.release()
    
    # If not enough frames were extracted, it might indicate a problem
    if len(frames) != num_frames:
        print(f"Warning: Extracted {len(frames)} out of {num_frames} requested for {video_path}")

    return np.array(frames) 
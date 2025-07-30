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
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return np.array(frames) 
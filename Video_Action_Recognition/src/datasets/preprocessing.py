"""Frame preprocessing extracted from Notebook Cells 3, 13 and 17.

IMPORTANT: Model inputs remain uint8 in [0,255], RGB, [T,H,W,C].
The final MobileNet model does /255 and ImageNet normalization in forward().
"""
from pathlib import Path
import cv2
import numpy as np

NUM_CACHE_FRAMES = 16
FRAME_SIZE = (112, 112)  # OpenCV width, height
MODEL_FRAME_IDS = tuple(np.linspace(0, NUM_CACHE_FRAMES - 1, 8).astype(int).tolist())


def process_frame(frame: np.ndarray, frame_size=FRAME_SIZE) -> np.ndarray:
    """Resize an RGB frame with INTER_AREA, preserving uint8 pixel values."""
    if frame is None:
        raise ValueError("Empty frame")
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB, RGBA or grayscale frame; got {frame.shape}")
    return cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA).astype(np.uint8)


def extract_frames(video_path, num_frames=NUM_CACHE_FRAMES, frame_size=FRAME_SIZE):
    """Original Cell 3 training-cache extractor (including its fallback rules)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Invalid video: {video_path}")
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        last_frame = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(rgb, frame_size, interpolation=cv2.INTER_AREA)
                last_frame = frame
            elif last_frame is not None:
                frame = last_frame.copy()
            else:
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            frames.append(frame)
        return np.stack(frames, axis=0)
    finally:
        cap.release()


def extract_video_frames(video_path, num_frames=NUM_CACHE_FRAMES):
    """Cell 17 demo extractor: sample 16 uniformly spaced frames; fallback if seeking fails.

    Unlike the cache extractor, supports streams with unreliable frame counts.
    Ordinary MP4 inputs with working seek follow the original Cell 17 path.
    """
    if isinstance(video_path, tuple):
        video_path = video_path[0]
    if not video_path:
        raise ValueError("Please upload or record a video")
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled = []
        if total > 0:
            positions = np.linspace(0, total - 1, num_frames).astype(int)
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ok, frame = cap.read()
                if not ok:
                    break
                sampled.append(process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(sampled) == num_frames:
                return np.stack(sampled)
    finally:
        cap.release()

    # Robust fallback for videos whose reported frame count or seeking is unreliable.
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot reopen video file: {path}")
    try:
        all_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if not all_frames:
        raise ValueError("No readable video frames")
    indices = np.linspace(0, len(all_frames) - 1, num_frames).astype(int)
    return np.stack([all_frames[int(i)] for i in indices])


def select_model_frames(frames: np.ndarray) -> np.ndarray:
    """Select [0,2,4,6,8,10,12,15] of 16 RGB frames -> [8,112,112,3]."""
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[0] < NUM_CACHE_FRAMES or arr.shape[-1] != 3:
        raise ValueError(f"Expected >=16 RGB frames [T,H,W,3], got {arr.shape}")
    return np.ascontiguousarray(arr[list(MODEL_FRAME_IDS)], dtype=np.uint8)

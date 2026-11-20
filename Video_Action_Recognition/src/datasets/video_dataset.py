"""Datasets extracted from Notebook Cells 6, 13 and 14.

The sample indices refer to the same order as data/cached_sample_order.csv.
Never shuffle the cache itself without regenerating matching split indices.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import MODEL_FRAME_IDS


class VideoDataset(Dataset):
    """3D CNN inputs, original Cell 6: [3,16,112,112], float in [0,1]."""
    def __init__(self, X, y, indices, augment=False):
        self.X, self.y = X, y
        self.indices = np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        video_id = int(self.indices[index])
        video = np.array(self.X[video_id], copy=True)
        label = int(self.y[video_id])
        if self.augment and np.random.rand() < 0.5:
            video = video[:, :, ::-1, :].copy()
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float() / 255.0
        return video, label


class VideoScratchDataset(Dataset):
    """Final full-MobileNet inputs, original Cell 13: [8,112,112,3], uint8."""
    def __init__(self, X, y, indices):
        self.X, self.y = X, y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx = int(self.indices[index])
        frames = np.ascontiguousarray(self.X[idx][list(MODEL_FRAME_IDS)])
        return torch.from_numpy(frames), torch.tensor(int(self.y[idx]), dtype=torch.long)


class AugmentedVideoDataset(VideoScratchDataset):
    """Training-only augmentation from Cell 14 (never use on validation/test)."""
    def __getitem__(self, index):
        idx = int(self.indices[index])
        frames = np.ascontiguousarray(self.X[idx][list(MODEL_FRAME_IDS)]).copy()
        if np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :].copy()
        brightness = np.random.uniform(0.9, 1.1)
        contrast = np.random.uniform(0.9, 1.1)
        frames = frames.astype(np.float32)
        frames = ((frames - 127.5) * contrast + 127.5) * brightness
        frames = np.clip(frames, 0, 255).astype(np.uint8)
        frames = np.ascontiguousarray(frames)
        return torch.from_numpy(frames), torch.tensor(int(self.y[idx]), dtype=torch.long)


def load_cached_arrays(data_dir):
    """Load the original 405-video cache and labels without reordering."""
    data_dir = Path(data_dir)
    cache = np.load(data_dir / "ucf101_16frames_112x112.npy", mmap_mode="r")
    import pandas as pd
    metadata = pd.read_csv(data_dir / "cached_sample_order.csv")
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    if cache.shape != (405, 16, 112, 112, 3) or labels.shape != (405,):
        raise ValueError("Cache / sample-order mismatch; expected (405,16,112,112,3)")
    return cache, labels


def load_split_indices(data_dir):
    """Return precomputed train/val/test splits from Cell 4, with leakage check."""
    data_dir = Path(data_dir)
    split = {name: np.load(data_dir / f"{name}_indices.npy")
             for name in ("train", "val", "test")}
    if any(len(set(split[a].tolist()) & set(split[b].tolist()))
           for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Data split index overlap detected")
    if [len(split[name]) for name in ("train", "val", "test")] != [300, 30, 75]:
        raise ValueError("Expected 300/30/75 split from notebook")
    return split

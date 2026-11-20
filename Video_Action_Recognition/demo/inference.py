"""Local inference with the exact pretrained MobileNetV3 final architecture.

Original notebook Cell 17 used this same architecture and selected 8 of 16 frames.
Place the REAL trained .pt checkpoint in checkpoints/. No model is trained here.
"""
import time
from pathlib import Path

import numpy as np
import torch
from src.models.mobilenet_temporal import FullMobileNetV3
from src.datasets.preprocessing import select_model_frames

CLASSES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
    "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
    "BasketballDunk", "BenchPress",
]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "full_mobilenet_pretrained_seed_2026.pt"


def find_state_dict(obj):
    """Read common checkpoint layouts without unsafe pickle loading."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model", "weights"):
            value = obj.get(key)
            if isinstance(value, dict) and any(isinstance(v, torch.Tensor) for v in value.values()):
                return value
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("No model state_dict found in checkpoint")


def load_model(checkpoint_path=DEFAULT_CHECKPOINT, device=None):
    """Load the user's actual trained .pt model; validate class count & frame layout."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Copy full_mobilenet_pretrained_seed_2026.pt from Google Drive/checkpoints."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        if "num_classes" in checkpoint and int(checkpoint["num_classes"]) != len(CLASSES):
            raise ValueError("Checkpoint class count is not 10")
        if "num_frames" in checkpoint and int(checkpoint["num_frames"]) != 8:
            raise ValueError("Checkpoint was trained with a different number of frames")
        if "frame_indices" in checkpoint:
            frame_ids = np.asarray(checkpoint["frame_indices"], dtype=int).tolist()
            if frame_ids != [0, 2, 4, 6, 8, 10, 12, 15]:
                raise ValueError(f"Checkpoint frame indices differ from the notebook: {frame_ids}")
    state = find_state_dict(checkpoint)
    state = {
        k.removeprefix("module.").removeprefix("_orig_mod."): v
        for k, v in state.items()
    }
    model = FullMobileNetV3(num_classes=len(CLASSES))
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [k for k in missing if k not in ("mean", "std")]
    if critical_missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            f"Missing: {critical_missing[:10]}, unexpected: {unexpected[:10]}"
        )
    return model.to(device).eval(), device


class ActionPredictor:
    """Single reusable model: avoids reloading weights on each Gradio request."""
    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device=None):
        self.model, self.device = load_model(checkpoint_path, device)

    @torch.inference_mode()
    def predict_frames(self, frames):
        selected = select_model_frames(np.asarray(frames))
        tensor = torch.from_numpy(selected).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        elapsed_ms = (time.perf_counter() - start) * 1000
        idx = int(np.argmax(probabilities))
        return {
            "action": CLASSES[idx],
            "confidence": float(probabilities[idx]),
            "scores": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))},
            "inference_ms": elapsed_ms,
        }

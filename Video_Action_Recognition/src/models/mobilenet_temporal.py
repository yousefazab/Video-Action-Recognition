"""MobileNetV3 classifiers from Cells 10, 13, and 17.

FullMobileNetV3 is the exact FINAL inference architecture: 8 RGB uint8 frames.
MobileNetVideoFull is the same trained model family, with optional ImageNet initialization.
Neither class downloads weights unless explicitly constructed pretrained=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class TemporalPoolingClassifier(nn.Module):
    """Cell 10: classifier for cached [B,16,576] ImageNet frame features."""
    def __init__(self, input_dim=576, num_classes=10, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(torch.mean(x, dim=1))


class MobileNetVideoFull(nn.Module):
    """Cell 13 pretrained-vs-scratch training architecture."""
    def __init__(self, pretrained=False, num_classes=10):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features
        self.classifier = nn.Sequential(
            nn.Linear(576, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, videos):
        b, t, h, w, c = videos.shape  # [B,T,H,W,C]
        frames = videos.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        frames = frames.float() / 255.0
        frames = (frames - self.mean) / self.std
        features = self.encoder(frames)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        features = features.reshape(b, t, -1)
        temporal_features = features.mean(dim=1)
        return self.classifier(temporal_features)


class FullMobileNetV3(nn.Module):
    """Cell 17 final Gradio inference architecture; checkpoint-compatible with Cell 13."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = mobilenet_v3_small(weights=None).features
        self.classifier = nn.Sequential(
            nn.Linear(576, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, videos):
        b, t, h, w, c = videos.shape  # [B,T,H,W,C]
        frames = videos.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        frames = frames.float() / 255.0
        frames = (frames - self.mean) / self.std
        features = self.encoder(frames)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        features = features.reshape(b, t, -1)
        temporal_features = features.mean(dim=1)
        return self.classifier(temporal_features)

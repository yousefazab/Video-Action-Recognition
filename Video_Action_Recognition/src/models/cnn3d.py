"""Simple3DCNN — architecture from notebook Cell 6."""
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 96, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(96, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

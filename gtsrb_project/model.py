# Senior Engineer Note:
# This is a simple CNN for traffic sign classification. Start small.

import torch.nn as nn


class TrafficSignCNN(nn.Module):
    # This function builds the beginner CNN layers once when the model is created.
    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    # This function describes how one batch of images moves through the CNN.
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

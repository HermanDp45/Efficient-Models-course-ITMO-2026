# models.py
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 100)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.relu(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.relu(x)

        x = self.linear2(x)
        return x

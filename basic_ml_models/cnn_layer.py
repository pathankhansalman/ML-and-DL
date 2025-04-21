import torch
import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    def forward(self, x):
        return self.conv(x)

import torch
from torch import nn


class SugarPCGroupEncoder(nn.Module):  # Embedding module
    def __init__(self, in_channels, output_size):
        super().__init__()
        self.in_channels = in_channels
        self.output_size = output_size

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.output_size, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, dim = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, dim)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # (b*g, 256, n)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (b*g, 256, 1)
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # (b*g, 512, n)
        feature = self.second_conv(feature)  # (b*g, output_size, n)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (b*g, output_size)
        return feature_global.reshape(bs, g, self.output_size)

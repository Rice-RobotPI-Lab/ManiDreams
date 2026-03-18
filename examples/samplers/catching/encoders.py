"""
Point cloud and observation encoders for RL.

Migrated from plate_catcher_demo/models/pointcloudae.py

Contains:
- PointCloudAE: PointNet-based AutoEncoder for point cloud encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudAE(nn.Module):
    """
    PointNet AutoEncoder for learning point cloud representations.

    Based on "Learning Representations and Generative Models For 3D Point Clouds"
    https://arxiv.org/abs/1707.02392

    Used for encoding Domain-Randomized State Sets (DRIS) into a fixed-size
    latent representation for policy learning.
    """

    def __init__(self, obs_dim=6, latent_dim=128, point_size=200):
        """
        Initialize PointCloudAE.

        Args:
            obs_dim: Dimension of each point (e.g., 6 for pos + vel)
            latent_dim: Dimension of the latent representation
            point_size: Number of points in the point cloud
        """
        super(PointCloudAE, self).__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.point_size = point_size

        # Encoder: PointNet-style 1D convolutions
        self.conv1 = torch.nn.Conv1d(self.obs_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_dim)

        # Decoder: MLP to reconstruct point cloud
        self.dec1 = nn.Linear(self.latent_dim, 256)
        self.dec2 = nn.Linear(256, 256)
        self.dec3 = nn.Linear(256, self.point_size * self.obs_dim)

    def encode(self, x):
        """
        Encode point cloud to latent representation.

        Args:
            x: Point cloud tensor of shape (B, N, D) where
               B = batch size, N = num points, D = obs_dim

        Returns:
            Latent representation of shape (B, latent_dim)
        """
        x = x.permute(0, 2, 1)  # convert shape (B, N, D) to (B, D, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # max pooling over points
        x = x.view(-1, self.latent_dim)
        return x

    def decode(self, x):
        """
        Decode latent representation to point cloud.

        Args:
            x: Latent tensor of shape (B, latent_dim)

        Returns:
            Reconstructed point cloud of shape (B, N, D)
        """
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, self.obs_dim)

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Args:
            x: Point cloud tensor of shape (B, N, D)

        Returns:
            Reconstructed point cloud of shape (B, N, D)
        """
        x = self.encode(x)
        x = self.decode(x)
        return x

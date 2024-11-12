import torch
import torch.nn as nn

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder= nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 784),
        )

    def forward(self, z):
        x = self.decoder(z)
        x = x.view(-1, 28, 28)
        return x
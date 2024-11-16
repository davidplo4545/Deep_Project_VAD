import torch
import torch.nn as nn

class VariationalAutoDecoder_LogNormal(nn.Module):
    def __init__(self, latent_dim, device="cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        self.decoder= nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 784),
        )

    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(mu).to(self.device)
        return torch.exp(epsilon * std + mu)

        
    def forward(self, distr_params):
        mu = distr_params[:, 0, :]
        std = distr_params[:, 1, :]
        x = self.reparameterize(mu, std)
        x = self.decoder(x)
        x = x.view(-1, 28, 28)
        return x

    def david_forward(self, z):
        x = self.decoder(z)
        x = x.view(-1, 28, 28)
        return x
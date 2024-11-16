import torch
import torch.nn as nn

class VariationalAutoDecoder_Expo(nn.Module):
    def __init__(self, latent_dim, device="cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        # self.mu_fc = nn.Linear(self.latent_dim, self.latent_dim)
        # self.std_fc = nn.Linear(self.latent_dim, self.latent_dim)
        
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

        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, 7 * 7 * 256),
        #     nn.ReLU(),
        #     nn.Unflatten(1, (256, 7, 7)),
        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)
        # )


    def reparameterize(self, rate):

        #directly making an Exp(1)
        # Sample noise from Exponential(1) distribution
        epsilon = torch.distributions.Exponential(1.0).sample(rate.shape).to(rate.device)

        #Manually making an Exp(1) using Uniform distrubtion
        # Sample uniformly from U(0, 1)
        # u = torch.rand_like(rate)
        # # Apply the reparameterization: X = -log(U)
        # epsilon = -torch.log(u)

        # Reparameterize: Z = (1 / rate) * epsilon
        z = (1 / rate) * epsilon
        return z

        
    def forward(self, distr_params):
        # mu = self.mu_fc(z)
        # std = self.std_fc(z)
        x = self.reparameterize(distr_params)
        x = self.decoder(x)
        x = x.view(-1, 28, 28)
        return x

    def david_forward(self, z):
        x = self.decoder(z)
        x = x.view(-1, 28, 28)
        return x
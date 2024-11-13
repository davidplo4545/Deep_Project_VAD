from evaluate import reconstruction_loss
import torch
import torch.nn as nn
import torch.optim as optim

class BasicTrainer:
    def __init__(self, model, dl, latent_dim=64, device='cpu'):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device
        self.latents = torch.nn.Parameter(torch.randn(len(self.dataloader.dataset), self.latent_dim).to(self.device))

        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents], lr=1e-3)
        self.loss = reconstruction_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (_, x) in enumerate(self.dataloader):
            samples = x.to(self.device)
            batch_size = samples.size(0)
            z = self.latents[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            reconstructed_z = self.model(z)

            loss = self.loss(samples, reconstructed_z)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        # Average loss over the epoch
        epoch_loss = running_loss / len(self.dataloader)
        return epoch_loss

    def train(self, num_epochs, early_stopping=None):
        """
        The training loop.
        """
        losses=list()
        best_loss = None
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            if best_loss is None or epoch_loss < best_loss:
                no_improvement = 0
                best_loss = epoch_loss
            else:
                no_improvement += 1
                if early_stopping is not None and no_improvement >= early_stopping:
                    break

        return losses


class VADTrainer:
    def __init__(self, model, dl, latent_dim=64, device='cpu'):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device
        
        self.mu = torch.randn(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)
        self.logvar = torch.randn(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)

        self.latents = torch.nn.parameter.Parameter(torch.stack([self.mu, self.logvar], dim=1)).to(device)

        # self.latents = torch.randn((len(self.dataloader.dataset), self.latent_dim)).to(self.device)

        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents] , lr=1e-3)
        # self.optimizer = optim.Adam(list(self.model.parameters()), lr=1e-3)


    # def trick(self, batch_mu, batch_logvar):
    #     sigma = torch.exp(0.5 * batch_logvar)
    #     epsilon = torch.randn_like(sigma)
    #     vec = batch_mu + sigma * epsilon
    #     return vec

    # def elbo_loss(self, orig, reco, batch_mu, batch_logvar):
    #     reco_loss = reconstruction_loss(orig, reco)
    #     kl_loss = -0.5 * torch.sum(1 + batch_logvar - batch_mu.pow(2) - batch_logvar.exp())
    #     kl_loss = kl_loss.mean()
    #     return reco_loss + kl_loss

    # def elbo_loss(self, orig, reco, batch_mu, batch_logvar):
    #     reco_loss = reconstruction_loss(orig, reco)
    #     kl_loss = -0.5 * torch.sum(1 + batch_logvar - batch_mu.pow(2) - batch_logvar.exp(), dim=1)
    #     kl_loss = kl_loss.mean()  # Average over the batch
    #     return reco_loss + kl_loss

    def elbo_loss(self, x, x_rec, mu, sigma):
        batch_size = x.size(0)
        rec_loss = reconstruction_loss(x, x_rec)
        log_var = torch.log(sigma.pow(2))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - sigma.pow(2), dim=1)
        return rec_loss +  42 * kl_loss.mean()
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (i, x) in enumerate(self.dataloader):
            samples = x.to(self.device)
            batch_size = samples.size(0)
            z = self.latents[i,:,:]

            # batch_mu = self.mu[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            # batch_logvar = self.logvar[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            # z = self.trick(batch_mu , batch_logvar)
            
            reconstructed_z = self.model(z)

            loss = self.elbo_loss(samples, reconstructed_z, z[:,0,:], z[:,1,:])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        # Average loss over the epoch
        epoch_loss = running_loss / len(self.dataloader)
        return epoch_loss

    def train(self, num_epochs, early_stopping=None):
        """
        The training loop.
        """
        losses=list()
        best_loss = None
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch()
            losses.append(epoch_loss)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            if best_loss is None or epoch_loss < best_loss:
                no_improvement = 0
                best_loss = epoch_loss
            else:
                no_improvement += 1
                if early_stopping is not None and no_improvement >= early_stopping:
                    break

        return losses



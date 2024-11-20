from evaluate import reconstruction_loss
import torch
import torch.nn as nn
import torch.optim as optim


def mse_loss(x, x_rec):
    """
    :param x: the original images
    :param x_rec: the reconstructed images
    :return: the mean squared error reconstruction loss
    """
    # Calculate MSE between x and x_rec, normalized over all elements per batch
    return torch.mean((x - x_rec) ** 2)


class BasicTrainer:
    def __init__(self, model, dl, latent_dim=64, device='cpu'):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device
        self.latents = torch.nn.Parameter(torch.randn(len(self.dataloader.dataset), self.latent_dim).to(self.device))

        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents], lr=5e-3)
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
    def __init__(self, model, dl, latent_dim=64, device='cpu', beta=1):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device
        
        self.mu = torch.randn(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)
        self.logvar = torch.randn(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)

        self.latents = torch.nn.parameter.Parameter(torch.stack([self.mu, self.logvar], dim=1)).to(device)

        self.beta = beta
        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents] , lr=1e-3)

    def elbo_loss(self, x, x_rec, mu, sigma):
        batch_size = x.size(0)
        rec_loss = mse_loss(x, x_rec)
        log_var = torch.log(sigma.pow(2))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - sigma.pow(2), dim=1)
        return rec_loss +  self.beta * kl_loss.mean()
        
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

class VADTrainer_Expo:
    def __init__(self, model, dl, latent_dim=64, device='cpu'):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device

        #regular
        self.latents = torch.nn.parameter.Parameter(torch.rand((len(self.dataloader.dataset), self.latent_dim))).to(self.device)
        self.latents = self.latents.detach().requires_grad_()

        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents] , lr=1e-3)
        # self.optimizer = optim.Adam(list(self.model.parameters()), lr=1e-3)

    def elbo_loss(self, x, x_rec, rate, prior_rate = 1.0):
        batch_size = x.size(0)
        rec_loss = mse_loss(x, x_rec)
        kl_loss = torch.sum(torch.log(rate / prior_rate) + (prior_rate / rate) - 1)
        return rec_loss + kl_loss.mean()
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (i, x) in enumerate(self.dataloader):
            samples = x.to(self.device)
            batch_size = samples.size(0)
            z = self.latents[i,:]

            # batch_mu = self.mu[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            # batch_logvar = self.logvar[batch_idx * batch_size : (batch_idx + 1) * batch_size, :]
            # z = self.trick(batch_mu , batch_logvar)
            
            reconstructed_z = self.model(z)

            loss = self.elbo_loss(samples, reconstructed_z, z)
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


class VADTrainer_LogNormal:
    def __init__(self, model, dl, latent_dim=64, device='cpu'):
        self.model = model.to(device)
        self.latent_dim = latent_dim
        self.dataloader = dl
        self.device = device

        self.mu = torch.rand(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)
        self.b = torch.rand(len(self.dataloader.dataset), self.latent_dim, requires_grad=True).to(self.device)
        self.latents = torch.nn.parameter.Parameter(torch.stack([self.mu, self.b], dim=1)).to(device)
        self.optimizer = optim.Adam(list(self.model.parameters()) + [self.latents] , lr=1e-3)

    def elbo_loss(self, x, x_rec, mu, sigma):
        batch_size = x.size(0)
        rec_loss = mse_loss(x, x_rec)
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma*2) - mu*2 - sigma*2)
        return rec_loss + kl_loss.mean()
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (i, x) in enumerate(self.dataloader):
            
            samples = x.to(self.device)
            batch_size = samples.size(0)
            z = self.latents[i,:,:]
            
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









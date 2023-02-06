import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        

        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size = 3, stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size = 3, stride=(2, 2), padding=(1, 1)),
        )
        self.conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256

        self.fc = nn.Linear(self.conv_out_dim, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 2*latent_dim)
    
    def forward(self, x):

        x = Encoder.forward(self, x)
        x = self.fc2(x)
        mu,std = x.chunk(2, dim = 1)
        return mu,std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        
        self.deconvs = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.reshape(z.shape[0], *self.base_size)
        z = self.deconvs(z)
        return z




class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
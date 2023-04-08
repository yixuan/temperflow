import numpy as np
import torch
import torch.nn as nn

class MNIST_Encoder(nn.Module):
    __constants__ = ["latent_dim"]

    def __init__(self, latent_dim, base_channels=16):
        super().__init__()
        # Number of features
        self.latent_dim = latent_dim
        ch1 = base_channels
        ch2 = ch1 * 2
        num_features = ch2 * 7 * 7

        # Convolutional layers to extract features
        conv1 = nn.Conv2d(in_channels=1, out_channels=ch1, kernel_size=4, stride=1, padding=2)
        conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch1, kernel_size=4, stride=2, padding=1)
        conv3 = nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=4, stride=2, padding=1)
        self.features = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True),
            conv3,
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.mu = nn.Linear(in_features=num_features, out_features=latent_dim)
        self.logvar = nn.Linear(in_features=num_features, out_features=latent_dim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        features = self.features(x)
        mu = self.mu(features)
        logvar = self.logvar(features)
        return mu, logvar

    def export_numpy(self, path):
        params = []
        for p in self.parameters():
            param = p.detach().cpu().numpy()
            if p.ndim == 2:
                param = param.transpose()
            elif p.ndim == 4:
                param = param.transpose(2, 3, 1, 0)
            params.append(param)
        np.savez_compressed(path, *params)

class MNIST_Generator(nn.Module):
    __constants__ = ["ch2"]

    def __init__(self, latent_dim, base_channels=16):
        super().__init__()
        ch1 = base_channels
        ch2 = ch1 * 2
        features = ch2 * 7 * 7
        self.ch2 = ch2
        self.upsample = nn.Linear(in_features=latent_dim, out_features=features)

        conv1 = nn.ConvTranspose2d(in_channels=ch2, out_channels=ch1, kernel_size=4, stride=2, padding=1)
        conv2 = nn.ConvTranspose2d(in_channels=ch1, out_channels=ch1, kernel_size=4, stride=2, padding=1, output_padding=1)
        conv3 = nn.ConvTranspose2d(in_channels=ch1, out_channels=1, kernel_size=4, stride=1, padding=2)
        self.net = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True),
            conv3,
            nn.Flatten()
        )

    def forward(self, x, apply_sigmoid: bool = True):
        x = self.upsample(x).relu_().view(-1, self.ch2, 7, 7)
        x = self.net(x).clamp_(-10.0, 10.0)
        if apply_sigmoid is True:
            x = torch.sigmoid_(x)
        return x

    def export_numpy(self, path):
        params = []
        for p in self.parameters():
            param = p.detach().cpu().numpy()
            if p.ndim == 2:
                param = param.transpose()
            elif p.ndim == 4:
                param = param.transpose(2, 3, 1, 0)
            params.append(param)
        np.savez_compressed(path, *params)

# Discriminator using CNN
class MNIST_Discriminator(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        ch1 = base_channels
        ch2 = ch1 * 2
        features = ch2 * 7 * 7
        conv1 = nn.Conv2d(in_channels=1, out_channels=ch1, kernel_size=4, stride=1, padding=2)
        conv2 = nn.Conv2d(in_channels=ch1, out_channels=ch1, kernel_size=4, stride=2, padding=1)
        conv3 = nn.Conv2d(in_channels=ch1, out_channels=ch2, kernel_size=4, stride=2, padding=1)
        linear = nn.Linear(in_features=features, out_features=1)
        self.net = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            conv2,
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            conv3,
            nn.ReLU(inplace=True),
            nn.Flatten(),
            linear
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.net(x)
        return x

    def export_numpy(self, path):
        params = []
        for p in self.parameters():
            param = p.detach().cpu().numpy()
            if p.ndim == 2:
                param = param.transpose()
            elif p.ndim == 4:
                param = param.transpose(2, 3, 1, 0)
            params.append(param)
        np.savez_compressed(path, *params)

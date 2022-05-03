import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.BatchNorm1d(num_features=hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(in_features=hidden_dim // 2,
                                   out_features=hidden_dim // 4)
        self.fc_var = nn.Linear(in_features=hidden_dim // 2,
                                out_features=hidden_dim // 4)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim // 4,
                      out_features=hidden_dim // 2),
            nn.BatchNorm1d(num_features=hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=input_dim))

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc_mu(h1), self.fc_var(h1)

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        rep = self.reparametrize(mu, logvar)
        reconstructed = self.decode(rep)
        return rep, reconstructed, mu, logvar

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.BatchNorm1d(num_features=hidden_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=hidden_dim // 2,
                      out_features=hidden_dim // 4),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim // 4,
                      out_features=hidden_dim // 2), 
            nn.BatchNorm1d(num_features=hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
            nn.Tanh())

    def forward(self, features):
        rep = self.encoder(features)
        reconstructed = self.decoder(rep)
        return rep, reconstructed, None, None


if __name__ == "__main__":
    print(VAE(16, 16))
    # print(AE(16, 16))
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import padding


class ConvAE(nn.Module):
    def __init__(self):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),  # 8, 13
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),  # 8, 13
            nn.Conv2d(32, 8, 3, stride=2, padding=1),  # b, 3, 6
            nn.BatchNorm2d(8),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            # [128, 16, 8, 13]
            nn.ConvTranspose2d(8, 32, 3, stride=2, output_padding=(0,1)),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               output_padding=(0, 1)),  # [128, 16, 8, 13]
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            # [128, 8, 15, 33]
            nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=(1, 1)),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 1, 2, stride=2),  # b, 1, 28, 28
        )
        
        # self.conv1 = nn.ConvTranspose2d(
        #     8, 16, 3, stride=2, output_padding=(1, 0))  # 128, 16, 8, 13

        # self.conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2)  # [128, 17, 27]
        
        # self.conv3 = nn.ConvTranspose2d(8, 1, 2, stride=3, padding=(1,2))
        
    def forward(self, x):
        
        
        ae_rep = self.encoder(x)
        # print(ae_rep.shape)
        recon_x = self.decoder(ae_rep)

        # recon_x = self.conv1(ae_rep)
        # print(recon_x.shape)
        # recon_x = self.conv2(recon_x)
        # print(recon_x.shape)
        # recon_x = self.conv3(recon_x)
        # print(recon_x.shape)
        return ae_rep, recon_x, None, None


if __name__ == "__main__":
    net = ConvAE()
    x = torch.randn((128, 1, 48, 76))
    print(x.shape)
    ae_rep, recon_x, _, _ = net(x)
    print(ae_rep.shape)
    print(recon_x.shape)
    
    

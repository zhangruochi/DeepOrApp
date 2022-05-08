import torch
import torch.nn as nn
from .AutoEncoder import VAE, AE
from .convAE import ConvAE
from .layers import FSlayer, Convlayer, UnSqueezeLayer, SqueezeLayer, MLP, DenseResnet


class DeepOrNet(nn.Module):
    def __init__(self, input_dim, ae_hidden, fs_rep_hidden, num_features,
                 variational):
        super(DeepOrNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(3, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(3, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(3, 1), stride=(2, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1),

            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=(3, 1), stride=(2, 1)),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d((1, input_dim))
        )

        if variational:
            self.ae = ConvAE()
        else:
            self.ae = ConvAE()

        self.fs = FSlayer(input_dim, fs_rep_hidden, num_features)

        # self.tab_resnet_blks = DenseResnet(
        #     128 + num_features + 144, [128,128, 128], 0.3
        # )

        self.gru = torch.nn.GRU(input_size=76, hidden_size=128,
                                num_layers=1, batch_first=True, bidirectional=False)

        self.classifier = MLP(
            [128 + num_features + 48 + 12, 16],
            "relu",
            0,
            True,
            False,
            True,
        )
        # self.classifier = torch.nn.Sequential(
        #     nn.Linear(ae_hidden // 4 + num_features, classifer_hidden),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(classifer_hidden, 1)
        # )

        self.dropout = torch.nn.Dropout(p=0)

    def forward(self, x, demographic_data):

        span_x = x.unsqueeze(1)
        # autoencoder
        ae_rep, recon_x, mu, logvar = self.ae(span_x)
        ae_rep = ae_rep.view(ae_rep.shape[0], -1)
        recon_x = recon_x.squeeze()

        # feature extractor
        feature_x = self.feature_extractor(span_x)
        # print(feature_x.shape) #[1 ,76]
        feature_x = feature_x.view(feature_x.shape[0], -1)

        # feature selection
        fs_rep, fs_feature, f_index = self.fs(feature_x)
        lstm_output, _ = self.gru(x)

        # print(lstm_output[:, -1, :].shape, ae_rep.shape, fs_feature.shape)

        # [128 + 144 + 32]
        cat_feature = torch.cat(
            [lstm_output[:, -1, :], ae_rep, fs_feature, demographic_data], dim=-1)

        final_feature = self.dropout(cat_feature)

        # final_feature = self.tab_resnet_blks(cat_feature)  # [b, 144]
        pre_logits = self.classifier(final_feature)

        return pre_logits, (x, ae_rep, recon_x, fs_rep, f_index, mu, logvar)

    def init_weight(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, FSlayer):
                # nn.init.constant_(m.params_1.data, val=0.01)
                nn.init.xavier_uniform_(m.params_1.data)
                # nn.init.xavier_uniform_(m.params_2.data)
                nn.init.constant_(m.bias_1.data, val=0.1)
                # nn.init.constant_(m.bias_2.data, val=0.1)

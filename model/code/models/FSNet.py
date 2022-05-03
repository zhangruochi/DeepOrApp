import torch
import torch.nn as nn
from .AutoEncoder import VAE, AE
from .layers import FSlayer, Convlayer, UnSqueezeLayer, SqueezeLayer, MLP, DenseResnet



class FSInbanceNet(nn.Module):
    def __init__(self, input_dim, ae_hidden, fs_rep_hidden, num_features,
                 variational):
        super(FSInbanceNet, self).__init__()

        if variational:
            self.ae = VAE(input_dim, ae_hidden)
        else:
            self.ae = AE(input_dim, ae_hidden)
            
        self.fs = FSlayer(input_dim, fs_rep_hidden, num_features)

        self.tab_resnet_blks = DenseResnet(
            ae_hidden // 4 +
            num_features, [(ae_hidden // 4 + num_features), (ae_hidden // 4 + num_features)//2, (ae_hidden // 4 + num_features)//2], 0.2
        )

        self.classifier = MLP(
            [(ae_hidden // 4 + num_features) // 2, (ae_hidden // 4 +
                                               num_features) // 2],
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

    def forward(self, x):
        ae_rep, recon_x, mu, logvar = self.ae(x)
        fs_rep, fs_feature, f_index = self.fs(x)
        concat_feature = torch.cat([ae_rep, fs_feature], dim=-1)

        res_features = self.tab_resnet_blks(concat_feature)
        pre_logits = self.classifier(res_features)
        
        return pre_logits, (x, ae_rep, recon_x, fs_rep, f_index, mu, logvar)

    def init_weight(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, FSlayer):
                # nn.init.constant_(m.params_1.data, val=0.01)
                # nn.init.constant_(m.params_2.data, val=0.01)
                nn.init.xavier_uniform_(m.params_1.data)
                # nn.init.xavier_uniform_(m.params_2.data)
                nn.init.constant_(m.bias_1.data, val=0.1)
                # nn.init.constant_(m.bias_2.data, val=0.1)

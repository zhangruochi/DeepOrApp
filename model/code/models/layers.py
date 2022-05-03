from collections import OrderedDict
import torch
import torch.nn as nn

from typing import Optional, Union, List
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module


class FSlayer(nn.Module):
    def __init__(self, input_feature_dim, fs_rep_hidden, num_features):
        super(FSlayer, self).__init__()
        self.num_features = num_features
        self.params_1 = nn.Parameter(torch.rand(input_feature_dim,
                                                fs_rep_hidden * 4),
                                     requires_grad=True)
        # self.params_2 = nn.Parameter(torch.rand(256,
        #                                         128),
        #                              requires_grad=True)

        self.bias_1 = torch.nn.Parameter(torch.rand(fs_rep_hidden * 4),
                                         requires_grad=True)
        # self.bias_2 = torch.nn.Parameter(torch.rand(128),
        #                                  requires_grad=True)

        self.resnet_block = DenseResnet(
            fs_rep_hidden * 4, [fs_rep_hidden * 4,
                                fs_rep_hidden * 4, 48], 0
        )

    def selset_features(self, x):
        _, f_index = torch.topk(torch.sum(torch.square(self.params_1.data),
                                          axis=1),
                                k=self.num_features,
                                largest=True)
        fs_feature = torch.index_select(x, dim=-1, index=f_index)
        return fs_feature, f_index

    def forward(self, x):
        hidden_rep = torch.relu(torch.mm(x, self.params_1) + self.bias_1)
        # fs_rep = torch.tanh(torch.mm(hidden_rep, self.params_2) + self.bias_2)
        fs_rep = self.resnet_block(hidden_rep)
        fs_feature, f_index = self.selset_features(x)

        return fs_rep, fs_feature, f_index


class SqueezeLayer(nn.Module):
    def __init__(self, dim=1):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, self.dim)


class UnSqueezeLayer(nn.Module):
    def __init__(self, dim=1):
        super(UnSqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class Convlayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 pooling_size=2):
        super(Convlayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=pooling_size),
        )

    def forward(self, x):
        return self.layer(x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class REGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    if activation == "tanh":
        return nn.Tanh()
    if activation == "gelu":
        return nn.GELU()
    if activation == "geglu":
        return GEGLU()
    if activation == "reglu":
        return REGLU()
    if activation == "softplus":
        return nn.Softplus()


def dense_layer(
    inp: int,
    out: int,
    activation: str,
    p: float,
    bn: bool,
    linear_first: bool,
):
    # This is basically the LinBnDrop class at the fastai library
    if activation == "geglu":
        raise ValueError(
            "'geglu' activation is only used as 'transformer_activation' "
            "in transformer-based models"
        )
    act_fn = get_activation_fn(activation)
    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        activation: str,
        dropout: Optional[Union[float, List[float]]],
        batchnorm: bool,
        batchnorm_last: bool,
        linear_first: bool,
    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

        self.mlp.add_module(
            "output_layer", nn.Linear(d_hidden[i], 1)
        )

        

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)
        

class BasicBlock(nn.Module):
    # inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L37
    def __init__(self, inp: int, out: int, dropout: float = 0.0, resize: Module = None):
        super(BasicBlock, self).__init__()

        self.lin1 = nn.Linear(inp, out)
        self.bn1 = nn.BatchNorm1d(out)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if dropout > 0.0:
            self.dropout = True
            self.dp = nn.Dropout(dropout)
        else:
            self.dropout = False
        self.lin2 = nn.Linear(out, out)
        self.bn2 = nn.BatchNorm1d(out)
        self.resize = resize

    def forward(self, x):

        identity = x

        out = self.lin1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        if self.dropout:
            out = self.dp(out)

        out = self.lin2(out)
        out = self.bn2(out)

        if self.resize is not None:
            identity = self.resize(x)

        out += identity
        out = self.leaky_relu(out)

        return out


class DenseResnet(nn.Module):
    def __init__(self, input_dim: int, blocks_dims: List[int], dropout: float):
        super(DenseResnet, self).__init__()

        self.input_dim = input_dim
        self.blocks_dims = blocks_dims
        self.dropout = dropout

        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin1", nn.Linear(input_dim, blocks_dims[0])),
                        ("bn1", nn.BatchNorm1d(blocks_dims[0])),
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
            
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks_dims[i - 1], blocks_dims[i]),
                    nn.BatchNorm1d(blocks_dims[i]),
                )
            self.dense_resnet.add_module(
                "block_{}".format(i - 1),
                BasicBlock(blocks_dims[i - 1],
                           blocks_dims[i], dropout, resize),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.dense_resnet(X)

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_c, out_c, hidden_ratio=1, dropout=0.0):
        super().__init__()

        hidden_c = in_c * hidden_ratio
        layers = []

        layers.append(nn.Conv2d(in_c, hidden_c, 1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(hidden_c, out_c, 1, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

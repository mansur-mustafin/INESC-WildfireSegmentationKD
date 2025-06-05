import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_c, out_c, hidden_ratio = 1):
        super.__init__()

        hidden_c = in_c * hidden_ratio

        self.net = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)

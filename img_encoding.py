import torch
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.resConv3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = self.resConv3(x)
        return F.relu(out + residual)
    

class ImgEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # batch(16, 14, 14)
        self.resBlock1 = ResBlock(in_channels=1,
                                  out_channels=16,
                                  stride=2)
        # batch (4, 7, 7)
        self.resBlock2 = ResBlock(in_channels=16,
                                  out_channels=4,
                                  stride=2)
        # batch (1, 4, 4)
        self.resBlock3 = ResBlock(in_channels=4,
                                  out_channels=1,
                                  stride=2)
        self.fc = nn.Linear(in_features=16,
                            out_features=8)
        self.ln = nn.LayerNorm(normalized_shape=8)

    def forward(self, x):
        out = self.resBlock3(self.resBlock2(self.resBlock1(x)))
        out = out.view(out.size(0), -1)
        out = self.ln(self.fc(out))
        return out
    

if __name__ == '__main__':
    img = torch.randn(1, 1, 28, 28)
    model = ImgEncoding()
    out = model(img)
    # print(out.shape)
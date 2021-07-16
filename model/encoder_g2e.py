from model.sid import Down,DoubleConv
import torch.nn as nn



class Encoderg2e(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x4 = x4[:, :, :-1, :]
        x5 = self.d4(x4)

        return x5
import re
from turtle import down, forward
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True, act='relu', use_dropout=False):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, padding_mode='reflect', bias=False)
            if is_down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2),
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        ) # 128
        
        self.down1 = Block(in_channels=features, out_channels=features*2, is_down=True, act='leaky', use_dropout=False) 
        self.down2 = Block(in_channels=features*2, out_channels=features*4, is_down=True, act='leaky', use_dropout=False) 
        self.down3 = Block(in_channels=features*4, out_channels=features*8, is_down=True, act='leaky', use_dropout=False)
        self.down4 = Block(in_channels=features*8, out_channels=features*8, is_down=True, act='leaky', use_dropout=False)
        self.down5 = Block(in_channels=features*8, out_channels=features*8, is_down=True, act='leaky', use_dropout=False)
        self.down6 = Block(in_channels=features*8, out_channels=features*8, is_down=True, act='leaky', use_dropout=False) 
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.3)
        ) # 1x1
        
        self.up1 = Block(features*8, features*8, is_down=False, act='relu', use_dropout=True)
        self.up2 = Block(features*8*2, features*8, is_down=False, act='relu', use_dropout=True)
        self.up3 = Block(features*8*2, features*8, is_down=False, act='relu', use_dropout=True)
        self.up4 = Block(features*8*2, features*8, is_down=False, act='relu', use_dropout=True)
        self.up5 = Block(features*8*2, features*4, is_down=False, act='relu', use_dropout=False)
        self.up6 = Block(features*4*2, features*2, is_down=False, act='relu', use_dropout=False)
        self.up7 = Block(features*2*2, features, is_down=False, act='relu', use_dropout=False)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
            d1 = self.initial_down(x) # 128
            d2 = self.down1(d1) # 64
            d3 = self.down2(d2) # 32
            d4 = self.down3(d3) # 16
            d5 = self.down4(d4) # 8
            d6 = self.down5(d5) # 4
            d7 = self.down6(d6) # 2
            bottlneck = self.bottleneck(d7) # 1
            
            up1 = self.up1(bottlneck) # 2
            up2 = self.up2(torch.cat([up1, d7], dim=1)) # 4
            up3 = self.up3(torch.cat([up2, d6], dim=1)) # 8
            up4 = self.up4(torch.cat([up3, d5], dim=1)) # 16
            up5 = self.up5(torch.cat([up4, d4], dim=1)) # 32
            up6 = self.up6(torch.cat([up5, d3], dim=1)) # 64
            up7 = self.up7(torch.cat([up6, d2], dim=1)) # 128
            return self.final_up(torch.cat([up7, d1], dim=1)) # 256
        
        
        
def test():
    x = torch.rand(1, 3, 256, 256)
    model = Generator(in_channels=3, features=64)
    pred = model(x)
    return pred.size()

if __name__ == '__main__':
    print(test())
    
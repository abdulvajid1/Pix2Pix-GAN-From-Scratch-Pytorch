from ast import mod
from pyexpat import model
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.block(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # we pass two images due condition image x, y -> concat across channel, so inchannel*2
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect')
        )
        
        layers = []
        
        for i in range(len(features) - 1): # - 1, features[i+1] cause out of list
            layers.append(
                CNNBlock(features[i], features[i+1], kernel_size=4, stride=1 if features[i+1]==features[-1] else 2)
                )
        
        layers.append(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
            
        self.model = nn.Sequential(*layers)
        
    
    def forward(self, x, target):
        x = torch.cat([x, target], dim=1) # target will be new channel in x
        x = self.initial(x)
        result = self.model(x)
        print(result.size())
    
    
def test():
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    model = Discriminator()
    output = model(x, y)
    print(output)
    

if __name__ == '__main__':
    test()
    
                
        
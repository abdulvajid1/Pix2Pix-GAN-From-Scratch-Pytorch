import torch
import config
from generator_model import Generator
from torchvision.utils import save_image


def save_some_samples(gen: Generator, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
    
    
    
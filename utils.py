from ast import mod
import torch
import config
from generator_model import Generator
from torchvision.utils import save_image


def save_some_samples(gen: Generator, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x) * 0.5 + 0.5
        save_image(y_fake, folder+f"/y_gen_{epoch}.png")
        save_image(x, folder+f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f'/label_{epoch}.png')
        gen.train()
        
def save_checkpoint(model, optimizer, filename='my_checkpoint.ckpt'):
    print("=> Saving Checkpoint")
    
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    
    torch.save(checkpoint, filename)
    
def load_checkpoint(file, model, optimizer, lr):
    checkpoint = torch.load(file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    
    
    
    
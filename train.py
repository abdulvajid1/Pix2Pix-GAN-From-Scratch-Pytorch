from networkx import desargues_graph
from pydantic import conint
from sympy import beta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint, save_some_samples, load_checkpoint
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/gan_experiment")

def train_step(generator, discriminator, optim_gen, optim_desc, g_scaler, d_scaler, train_loader, bce_loss, l1_loss, val_loader):
    loop = tqdm(train_loader, leave=True)
    
    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        
        # print('train loop 1 is ok')
        # print("input =>",x.dtype)
        # print("y dtype =>",y.dtype)
        # print("input shape", x.shape)
        # print("y shape", y.shape)
        
        with torch.amp.autocast(device_type=config.DEVICE):
            y_fake = generator(x)
            D_real = discriminator(x, y)
            D_fake = discriminator(x, y_fake.detach())
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2 
            
        optim_desc.zero_grad()    
        d_scaler.scale(D_loss).backward()
        d_scaler.step(optim_desc)
        d_scaler.update()
        
        
        with torch.amp.autocast(device_type=config.DEVICE):
            y_fake = generator(x)
            D_fake = discriminator(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            l1 = l1_loss(y, y_fake)
            G_loss = G_fake_loss + l1 * config.L1_LAMBDA
            
        optim_gen.zero_grad()    
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optim_gen)
        g_scaler.update()
        
        if idx % 50 == 0:
            save_some_samples(generator, val_loader, idx, folder='eval_files')
        
        loop.set_postfix({
        "D_loss": D_loss.item(),
        "G_loss": G_loss.item()
        })
        
    return D_loss.item(), G_loss.item()
        

def main():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3).to(config.DEVICE)
    optim_desc = optim.AdamW(discriminator.parameters(), lr=config.LR, betas=(0.5, 0.999))
    optim_gen = optim.AdamW(generator.parameters(), lr=config.LR, betas=(0.5, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    if config.LOAD_MODEL:
        print('Loadin Checkpoint')
        load_checkpoint(config.GEN_CHECKPOINT_PATH, generator, optim_gen, lr=config.LR)
        load_checkpoint(config.DESC_CHECKPOINT_PATH, discriminator, optim_desc, lr=config.LR)
        
    train_dataset = MapDataset(root_dir='dataset/train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    
    val_dataset = MapDataset(root_dir='dataset/val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()
    
    
    for epoch in range(config.NUM_EPOC):
        D_loss, G_loss = train_step(generator, discriminator, optim_gen, optim_desc, g_scaler, d_scaler, train_loader, bce_loss, l1_loss, val_loader)
        if writer:
            writer.add_scalar("Loss/Discriminator", D_loss, epoch)
            writer.add_scalar("Loss/Generator", G_loss, epoch)
        
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(generator, optim_gen, config.GEN_CHECKPOINT_PATH)
            save_checkpoint(discriminator, optim_desc, config.DESC_CHECKPOINT_PATH)
        
        save_some_samples(generator, val_loader, epoch, folder='eval_files')
        
        
if __name__ == "__main__":
    main()

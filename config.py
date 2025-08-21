from pickle import TRUE
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-5
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
IMG_CHANNEL = 3
L1_LAMBDA = 100
NUM_EPOC = 100
LAMBDA_GP = 10
LOAD_MODEL = True
SAVE_MODEL = True

DESC_CHECKPOINT_PATH = 'desc_pth.ckpt'
GEN_CHECKPOINT_PATH = 'gen_pth.ckpt'

both_transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(0.5),
], additional_targets=['image0' : 'image'])

tranform_input = A.Compose([
    A.ColorJitter(p=0.1)
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2()
])

tranform_target = A.Compose([
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2()
])
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_dir:str = 'dataset'):
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        image_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, image_file)
        image = np.array(Image.open(img_path))
        
        print(f"Image size before splitting {image.shape}")
        
        input_img = image[:, :600, :] # split the image input and target to image (1200, height) -> (600, h) , (600, h)
        target_img = image[:, 600:, :]
        
        print(f"Image size after splitting {input_img.shape}")
        
        augmentation = config.both_transform(image=input_img, image0=target_img)
        input_img, target_img = augmentation['image'], augmentation['image0']
        
        input_img = config.tranform_input(image=input_img)['image']
        target_img = config.tranform_target(image=target_img)['image']
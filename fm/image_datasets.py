import random

import cv2
                        
import numpy as np
from torch.utils.data import Dataset
import os
join = os.path.join
EXTs = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

                                              
                                              
                                              
def imread_uint(path, n_channels=3):
                  
                                             
    if n_channels == 1:
        img = cv2.imread(path, 0)                        
        img = np.expand_dims(img, axis=2)         
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)            
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)       
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
    return img

class ImageDataset(Dataset):
    def __init__(
        self,
        config,
        phase="train",
    ):
        super().__init__()
        self.config = config
        self.phase = phase
        self.img_size = config.img_size                                       
        image_paths = config.train_path if phase == "train" else config.val_path
        image_paths_list = [image_paths] if isinstance(image_paths, str) else image_paths
        recursive = getattr(config, "recursive", True)
        self.image_names = []
        for image_path in image_paths_list:
            if not os.path.exists(image_path):
                continue
            if recursive:
                for root, _, files in os.walk(image_path):
                    self.image_names.extend([os.path.join(root, image_name) for image_name in sorted(files)])
            else:
                self.image_names.extend([os.path.join(image_path, image_name) for image_name in sorted(os.listdir(image_path))])
        self.image_names = [image_name for image_name in self.image_names if os.path.splitext(image_name)[-1].lower() in EXTs]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        path = self.image_names[idx]
        img_H = imread_uint(path, self.config.in_channels)
                      
        if self.config.resize_size > 0:                                                 
            height, width, _ = img_H.shape
            if height < width:
                new_height = self.config.resize_size
                new_width = int((width / height) * new_height)
            else:
                new_width = self.config.resize_size
                new_height = int((height / width) * new_width)
                                                                                               
                                                             
            img_H = cv2.resize(img_H, (new_width, new_height), interpolation=cv2.INTER_AREA)
        H, W, _ = img_H.shape
        if self.phase == "train":
            if self.config.random_crop:                    
                rnd_h = random.randint(0, max(0, H - self.img_size))
                rnd_w = random.randint(0, max(0, W - self.img_size))
                arr = img_H[rnd_h:rnd_h + self.img_size, rnd_w:rnd_w + self.img_size, :]
            elif self.config.center_crop:              
                arr = img_H[(H - self.img_size) // 2:(H + self.img_size) // 2, 
                            (W - self.img_size) // 2:(W + self.img_size) // 2, :]
            else:            
                arr = img_H
        elif self.phase == "val":
            if self.config.random_crop_val:                    
                rnd_h = random.randint(0, max(0, H - self.config.val_img_size))
                rnd_w = random.randint(0, max(0, W - self.config.val_img_size))
                arr = img_H[rnd_h:rnd_h + self.config.val_img_size, rnd_w:rnd_w + self.config.val_img_size, :]
            elif self.config.center_crop_val:              
                arr = img_H[(H - self.config.val_img_size) // 2:(H + self.config.val_img_size) // 2, 
                            (W - self.config.val_img_size) // 2:(W + self.config.val_img_size) // 2, :]
            else:            
                arr = img_H
        else:
            raise NotImplementedError
                                                        

        arr = arr.astype(np.float32) / 127.5 - 1          
        out_dict = {'img_name': os.path.splitext(os.path.basename(self.image_names[idx]))[0]}
                                     
        return np.transpose(arr, [2, 0, 1]), out_dict


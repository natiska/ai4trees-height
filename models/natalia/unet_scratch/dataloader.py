import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image 
import numpy as np

class segDataset(Dataset):
   def __init__(self,img_dir,mask_dir,transform):
      self.img_dir = img_dir
      self.mask_dir = mask_dir
      self.transform = transform
      self.image_list = os.listdir(img_dir)
      self.masks_list= os.listdir(mask_dir)
    
    
   def __len__(self):
       return len(self.image_list)
   
   def __getitem__(self,index):
       img = np.array(Image.open(os.path.join(self.img_dir,self.image_list[index])))
       mask = np.array(Image.open(os.path.join(self.mask_dir,self.image_list[index])).convert("L"),dtype=np.float32)
       mask[mask==255.0] = 1.0
       augmentations =self.transform(image=img,mask=mask)
       img=augmentations["image"]
       mask=augmentations["mask"]
       filename=self.image_list[index]
       
       return img,mask,filename
       
def get_loader(img_dir,mask_dir,train_transform,test_transform,batch_size,shuffle):
    train_dataset = segDataset(os.path.join(img_dir, "train"),os.path.join(mask_dir, "train"),train_transform)
    valid_dataset = segDataset(os.path.join(img_dir, "validation"),os.path.join(mask_dir, "validation"),test_transform)
    test_dataset = segDataset(os.path.join(img_dir, "test"),os.path.join(mask_dir, "test"),test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=shuffle)
    
    return train_loader, valid_loader, test_loader



import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset , Subset
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
       mask = np.array(Image.open(os.path.join(self.mask_dir,self.image_list[index].replace(".jpg","_mask.gif"))).convert("L"),dtype=np.float32)
       mask[mask==255.0] = 1.0
       augmentations =self.transform(image=img,mask=mask)
       img=augmentations["image"]
       mask=augmentations["mask"]
       
       return img,mask
       
def get_loader(img_dir,mask_dir,train_transform,test_transform,batch_size,shuffle):
    train_dataset = segDataset(img_dir,mask_dir,train_transform)
    test_dataset = segDataset(img_dir,mask_dir,test_transform)
    test_size=0.10
    seed=7777
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_size * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices,test_indices = indices[split:],indices[:split]
    
    # train_sampler = SubsetRandomSampler(train_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, 
    #                                        sampler=train_sampler)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size,
    #                                             sampler=test_sampler)
    
    train_dataset=Subset(train_dataset,train_indices)
    test_dataset = Subset(test_dataset,test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=shuffle)
    
    return train_loader,test_loader



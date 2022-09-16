import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob

class segDataset(Dataset):
    def __init__(self, data_dir, input_folders, output_folder, transform):
        self.data_dir = data_dir
        self.input_folders = input_folders
        self.output_folder = output_folder
        assert len(self.input_folders) > 0
        self.image_list = os.listdir(os.path.join(data_dir, output_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        #print(self.image_list[index])
        if len(self.input_folders) > 1:
            input_layers = []
            for folder_name in self.input_folders:
                array = np.array(Image.open(os.path.join(self.data_dir, folder_name, self.image_list[index])))
                if len(array.shape) == 2: # convert (256, 256) into (256, 256, 1)
                    prev_shape = array.shape
                    array = array.reshape(array.shape[0], array.shape[1], 1)
                if len(array.shape) == 4: # convert [1, 4, 256, 256] into [4, 256, 256]
                    array = array.squeeze()
                if (len(array.shape) == 3) and (4 in list(array.shape)):
                    channel_idx = list(array.shape).index(4)
                    if channel_idx == 0:
                        array = array[:-1,:,:]
                    else:
                        array = array[:,:,:-1]
                input_layers.append(array)
            array = np.dstack(input_layers)
        else:
            array = np.array(Image.open(os.path.join(
                self.data_dir, self.input_folders[0], self.image_list[index])))
            if len(array.shape) == 4:
                array = array.squeeze()
            if (len(array.shape) == 3) and (4 in list(array.shape)):
                channel_idx = list(array.shape).index(4)
                if channel_idx == 0:
                    array = array[:-1,:,:]
                else:
                    array = array[:,:,:-1]
        img = array.astype(np.float32)
        mask = np.array(Image.open(os.path.join(self.data_dir, self.output_folder,
                        self.image_list[index])).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0
        augmentations = self.transform(image=img, mask=mask)
        img = augmentations["image"]
        mask = augmentations["mask"]
        filename = self.image_list[index]

        return img, mask, filename

def get_loader(data_dir, input_folders, output_folder, train_transform, test_transform, batch_size, shuffle):
    train_dataset = segDataset(os.path.join(
        data_dir, "train"), input_folders, output_folder, train_transform)
    valid_dataset = segDataset(os.path.join(
        data_dir, "validation"), input_folders, output_folder, test_transform)
    test_dataset = segDataset(os.path.join(
        data_dir, "test"), input_folders, output_folder, test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader

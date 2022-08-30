import torch
import torch.nn as nn
import os
from PIL import Image 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
import segmentation_models_pytorch as smp
import yaml
import sys
from dataloader import *

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def access_drive():
    # https://www.wpdisplayfiles.com/docs/how-to-generate-google-drive-client-id-client-secret-key/
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    return drive

def make_train_test_transform(img_size, mean, std, max_pixel_value):
    train_transform = A.Compose(
                [A.Resize(img_size[0],img_size[1]),
                A.Rotate(limit=35,p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=max_pixel_value
                ),
                ToTensorV2(),
                ])
    test_transform = A.Compose(
                [A.Resize(img_size[0],img_size[1]),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=max_pixel_value
                ),
                ToTensorV2(),
                ])
    return train_transform, test_transform

def train_model(num_epochs, train_loader, model, optimizer, loss_fn, device):
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        train_loss=train_one_epoch(train_loader,model,optimizer,loss_fn,device)
        print(train_loss)
    return
    
def train_one_epoch(train_loader,model, optimizer,loss_fn,device):
    running_tloss = 0.0
    loop = tqdm(train_loader)
    model = model.to(device)
    model.train()
    for batch_idx, (img,mask,f) in enumerate(loop):
        img=img.to(device)
        mask= mask.float().unsqueeze(1).to(device)
        #print(mask.shape)
        predictions = model(img)
        #print(predictions.shape)
        loss = loss_fn(predictions,mask)
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        running_tloss = running_tloss + loss.item()
        loop.set_postfix(loss=loss.item())
        
    epoch_tloss=running_tloss/len(train_loader)
    return epoch_tloss
        
def evaluate(test_loader,model,loss_fn,device):
    dice_score = 0.0
    running_vloss = 0.0
    loop= tqdm(test_loader)
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx,(img,mask,f) in enumerate(loop):
            img=img.to(device)
            mask= mask.float().unsqueeze(1).to(device)
            predss = model(img)
            loss = loss_fn(predss,mask)
            predss = torch.sigmoid(predss)
            preds = (predss>0.5).float()
            dice_score=dice_score + (2 * (preds * mask).sum()) / (
                (preds + mask).sum() + 1e-8
            )
            running_vloss += loss.item()
        epoch_dice_score=dice_score/len(test_loader)
        epoch_vloss=running_vloss/len(test_loader)
    model.train()
    return epoch_dice_score.item(),epoch_vloss
            
def save_predictions_as_imgs(loader, model, device, folder="saved_images/"):
    model.eval()
    for batch_idx, (img,mask,fname) in enumerate(loader):
        img = img.to(device)
        mask= mask.float().unsqueeze(1).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
        save_image(preds, f"{folder}/pred_{batch_idx}_{fname}.png")
        save_image(mask, f"{folder}{batch_idx}.png")

def main(config):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transform, test_transform = make_train_test_transform(config["img_size"],
    config["mean"], config["std"], config["max_pixel_value"])

    train_loader,test_loader=get_loader(config["img_dir"],
                                        config["mask_dir"],
                                        train_transform,
                                        test_transform,
                                        config["batch_size"],
                                        shuffle=True)

    model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1,activation=None, encoder_depth=5, decoder_channels=[512, 256, 128,64,32])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])

    train_model(config.num_epochs, train_loader, model, optimizer, loss_fn, device)

    save_predictions_as_imgs(test_loader,model,device,"saved_images/")

if __name__ == '__main__':

    config_path = sys.argv[1]
    config = read_yaml(config_path)
    print(config)
    #main(config)
import torch
import torch.nn as nn
import os
from PIL import Image 
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image
import segmentation_models_pytorch as smp


###### PARAMS #######
#### CHANGE HERE ####
#####################
batch_size = 2
learning_rate = 1e-3
weight_decay = 1e-4
num_epochs = 5
imgsize=(256,256)
#encoder_kernlist=[3,64,128,256,512,1024]
#decoder_kernelist=[1024,512, 256, 128, 64]
img_dir="data/Tree_RGB"
mask_dir="data/Tree_masks"
###################
###################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###################################

train_transform = A.Compose(
    [
        
      A.Resize(imgsize[0],imgsize[1]),
      A.Rotate(limit=35,p=1.0),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.1),
      A.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
          max_pixel_value=255.0
      ),
      ToTensorV2(),
     ]
)



test_transform = A.Compose(
    [
        
      A.Resize(imgsize[0],imgsize[1]),
      A.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
          max_pixel_value=255.0
      ),
      ToTensorV2(),
     ]
)
 
 
########################################## Dataloader Verification ###############################################
# train_loader,test_loader=get_loader(img_dir,mask_dir,train_transform,test_transform,batch_size,shuffle=False)


# images,masks=next(iter(train_loader))


# rows = 4
# columns = 4
# masks=masks.unsqueeze(1)

# print(images.shape)
# print(masks.shape)
# fig,ax = plt.subplots(4,4,figsize = (30,20))
# ax = ax.ravel()
# j=0
# for i in range(0,16,2):
#     ax[i].imshow(images[j].numpy().transpose((1, 2, 0)))
#     ax[i+1].imshow(masks[j].numpy().transpose((1, 2, 0)))
#     j=j+1    
# plt.show()
##########################################################################################################################



train_loader,test_loader=get_loader(img_dir,mask_dir,train_transform,test_transform,batch_size,shuffle=True)

model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1,activation=None, encoder_depth=5, decoder_channels=[512, 256, 128,64,32])

loss_fn = nn.BCEWithLogitsLoss()


optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

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
        
############ one_epoch verification ####################
# c= train_one_epoch(train_loader,model,optimizer,loss_fn,device)
# print(c)
######################################################

def evaluate (test_loader,model,loss_fn,device):
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
            
############ Evaluate verification  ####################
# c= evaluate (test_loader,model,device)
# print(c)
######################################################     

def save_predictions_as_imgs(loader, model, folder="saved_images/", device=device):
    model.eval()
    for batch_idx, (img,mask,fname) in enumerate(loader):
        img = img.to(device)
        mask= mask.float().unsqueeze(1).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
        save_image(preds, f"{folder}/pred_{batch_idx}_{fname}.png")
        save_image(mask, f"{folder}{batch_idx}.png")



       
            
for epoch in range(num_epochs):
    print('EPOCH {}:'.format(epoch + 1))
    train_loss=train_one_epoch(train_loader,model,optimizer,loss_fn,device)
    print(train_loss)
    # test_dice,test_loss = evaluate (test_loader,model,loss_fn,device)
    # print('LOSS train {} valid {} Dice Score {}'.format(train_loss, test_loss , test_dice))
    
    
save_predictions_as_imgs(test_loader,model,"saved_images/",device)
    
    

    
    
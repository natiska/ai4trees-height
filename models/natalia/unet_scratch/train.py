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
from model_unet import UNET
import wandb

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def make_train_test_transform(img_size):
    
    img_size = tuple(img_size)
    
    train_transform = A.Compose(
                [A.Resize(img_size[0],img_size[1]),
                A.Rotate(limit=35,p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                #A.Normalize(
                #    mean=mean,
                #    std=std,
                #    max_pixel_value=max_pixel_value
                #),
                ToTensorV2(),
                ])
    test_transform = A.Compose(
                [A.Resize(img_size[0],img_size[1]),
                #A.Normalize(
                #    mean=mean,
                #    std=std,
                #    max_pixel_value=max_pixel_value
                #),
                ToTensorV2(),
                ])
    return train_transform, test_transform

def train_model(num_epochs, train_loader, valid_loader, model, optimizer, loss_fn, device):
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        train_loss=train_one_epoch(train_loader,model,optimizer,loss_fn,device)
        print("Train set loss: ", train_loss)
        wandb.log({"Train set loss":train_loss})
        valid_dice_score, valid_loss, v_accuracy, v_precision, v_recall, v_fscore = evaluate(valid_loader,model,loss_fn,device,send_to_wandb=False)
        print("Validation set loss: ", valid_loss)
        wandb.log({"Validation set loss": valid_loss})
        print("Validation set DICE score: ", valid_dice_score)
        wandb.log({"Validation set DICE score": valid_dice_score})
        print("Validation set accuracy: ", v_accuracy)
        wandb.log({"Validation set accuracy: ": v_accuracy})
        print("Validation set precision: ", v_precision)
        wandb.log({"Validation set precision: ": v_precision})
        print("Validation set recall: ", v_recall)
        wandb.log({"Validation set recall: ": v_recall})
        print("Validation set F-score: ", v_fscore)
        wandb.log({"Validation set F-score: ": v_fscore})
    return
    
def train_one_epoch(train_loader, model, optimizer,loss_fn,device):
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
    return np.round(epoch_tloss,3)

def compute_accuracy(prediction, ground_truth):
    assert prediction.shape == ground_truth.shape
    dims = list(prediction.shape)
    size = 1
    for dim in dims:
      size *= dim
    accuracy = np.round(float((prediction == ground_truth).sum())/size, 2)*100
    return accuracy

def compute_precision_recall_fscore(prediction, ground_truth):
    TP = int((ground_truth == prediction)[ground_truth==1].sum())
    TN = int((ground_truth == prediction)[ground_truth==0].sum())
    FP = int((ground_truth != prediction)[ground_truth==0].sum())
    FN = int((ground_truth == prediction)[ground_truth==1].sum())
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    fscore = (2 * precision * recall)/(precision + recall)
    return np.round(precision,2), np.round(recall,2), np.round(fscore,2)

def evaluate(test_loader,model,loss_fn,device, send_to_wandb):
    dice_score = 0.0
    running_vloss = 0.0
    overall_accuracy = 0.0
    overall_precision = 0.0
    overall_recall = 0.0
    overall_fscore = 0.0
    loop= tqdm(test_loader)
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx,(raw_img,raw_mask,f) in enumerate(loop):
            img=raw_img.to(device)
            mask=raw_mask.float().unsqueeze(1).to(device)
            predss = model(img)
            loss = loss_fn(predss,mask)
            predss = torch.sigmoid(predss)
            preds = (predss>0.5).float()
            dice_score=dice_score + (2 * (preds * mask).sum()) / (
                (preds + mask).sum() + 1e-8
            )
            running_vloss += loss.item()
            accuracy = compute_accuracy(preds, mask)
            overall_accuracy += accuracy
            precision, recall, fscore = compute_precision_recall_fscore(preds, mask)
            overall_precision += precision
            overall_recall += recall
            overall_fscore += fscore
            if send_to_wandb:
                log_preds_in_wandb(f, raw_img, raw_mask, preds)
        epoch_accuracy = overall_accuracy/len(test_loader)
        epoch_precision = np.round(overall_precision/len(test_loader),2)
        epoch_recall = np.round(overall_recall/len(test_loader),2)
        epoch_fscore = np.round(overall_fscore/len(test_loader),2)
        epoch_dice_score=dice_score/len(test_loader)
        epoch_vloss=np.round(running_vloss/len(test_loader),2)
    model.train()
    return np.round(epoch_dice_score.item(),2),epoch_vloss, epoch_accuracy, epoch_precision, epoch_recall, epoch_fscore
            
def save_predictions_as_imgs(loader, model, device, output_folder):
    model.eval()
    predictions_folder = output_folder + "/predictions"
    if os.path.exists(predictions_folder) == False:
        os.makedirs(predictions_folder)
    for batch_idx, (img,mask,fname) in enumerate(loader):
        img = img.to(device)
        mask= mask.float().unsqueeze(1).to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
        save_image(preds, f"{predictions_folder}/pred_{fname[0].split('.')[0]}.png")

def log_preds_in_wandb(filename, img, mask, predicted_mask):
        class_labels = {0: "no-tree", 1: "tree"}
        if img.shape[1] == 4: # the 4-th channel is height and we want to visualize it separately
            DSM = img[:,3,:,:]
            img = img[:,:3,:,:]
        else:
            DSM = None
        true_mask_array = np.squeeze(np.array(mask))
        pred_mask_array = np.squeeze(np.array(predicted_mask))
        RGB_image_with_masks = wandb.Image(img, masks={"predictions" : {
                                                    "mask_data" : pred_mask_array,
                                                    "class_labels" : class_labels},
                                                "ground_truth" : {
                                                    "mask_data" : true_mask_array,
                                                    "class_labels" : class_labels}})
        if DSM is None:
            wandb.log({filename[0]: RGB_image_with_masks})
        else:
            DSM_image = wandb.Image(DSM)
            wandb.log({filename[0]: [RGB_image_with_masks, DSM_image]})

def main(config):

    if bool(config["log_in_wandb"]) == False:
        os.environ['WANDB_DISABLED'] = 'true'

    wandb.init(project="ai4trees_project")
    wandb.config = config
    for key in config:
        wandb.log({key: config[key]})

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    train_transform, test_transform = make_train_test_transform(config["img_size"])

    train_loader, valid_loader, test_loader=get_loader(config["data_dir"],
                                        list(config["input_folders"]),
                                        config["output_folder"],
                                        train_transform,
                                        test_transform,
                                        config["batch_size"],
                                        shuffle=True)

    if bool(config["use_pretrained"]) and config["input_channels"] == 3:
        wandb.log({"Base": "ResNet34"})
        model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=1,
                        activation=None, encoder_depth=5, decoder_channels=[512, 256, 128,64,32])
    else:
        wandb.log({"Base": "-"})
        model = UNET(in_channels=config["input_channels"],out_channels=1, features=list(config["features"]))
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=config["learning_rate"])

    train_model(config["num_epochs"], train_loader, valid_loader, model, optimizer, loss_fn, device)

    test_dice_score, test_loss, t_accuracy, t_precision, t_recall, t_fscore = evaluate(test_loader,model,loss_fn,device, send_to_wandb=True)
    print("Test dice score: ", test_dice_score)
    wandb.log({"Test dice score": test_dice_score})
    print("Test loss: ", test_loss)
    wandb.log({"Test loss": test_loss})
    print("Test set accuracy: ", t_accuracy)
    wandb.log({"Test set accuracy: ": t_accuracy})
    print("Test set precision: ", t_precision)
    wandb.log({"Test set precision: ": t_precision})
    print("Test set recall: ", t_recall)
    wandb.log({"Test set recall: ": t_recall})
    print("Test set F-score: ", t_fscore)
    wandb.log({"Test set F-score: ": t_fscore})

    if bool(config["save_predictions"]):
        save_predictions_as_imgs(test_loader,model,device, config["save_output_path"])
    if bool(config["save_model"]):
        torch.save(model.state_dict(), os.path.join(config["save_output_path"], 'model_weights.pth'))

if __name__ == '__main__':

    config_path = sys.argv[1]
    config = read_yaml(config_path)
    main(config)
    print("Script finished")
from os import device_encoding
import torch
import torch.nn as nn
from modules.encoder import DoubleConv,segEncoder
from modules.decoder import segDecoder
 

class unet(nn.Module):
    def __init__(self,ch_list_en,ch_list_de,output_size,out_channels):
        super(unet,self).__init__()
        self.encoder=segEncoder(ch_list_en)
        self.decoder=segDecoder(ch_list_de)
        self.head=nn.Conv2d(ch_list_de[-1],out_channels,1)
        self.output_size=output_size
        
        
    def forward(self,x):
        enc_features = self.encoder(x)
        out=self.decoder(enc_features[::-1][0],enc_features[::-1][1:]) 
        out=self.head(out)
        out=nn.functional.interpolate(out,self.output_size)
        return out
    
##################################################
################### Verification #################    
    
# model=unet([3,64,128,256,512,1024],[1024,512, 256, 128, 64],(420,260),1) 
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model=model.to(device)
# x = torch.randn(2, 3,420,260)
# x=x.to(device)
# output=model(x)
# print(output.shape)

##################################################
        
        
        




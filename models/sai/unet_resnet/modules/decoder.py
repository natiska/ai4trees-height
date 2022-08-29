import torch
import torch.nn as nn
import torchvision
from modules.encoder import DoubleConv,segEncoder




class segDecoder(nn.Module):
    def __init__(self,ch_list):
        super(segDecoder,self).__init__()
        self.ch_list=ch_list
        self.ml=nn.ModuleList()
        self.ml2= nn.ModuleList()
        
        for i in range(0,len(ch_list)-1):
            self.ml.append(DoubleConv(ch_list[i],ch_list[i+1]))
            
        for i in range(0,len(ch_list)-1):
            self.ml2.append(nn.ConvTranspose2d(ch_list[i],ch_list[i+1],2,2))
            
  
    def forward(self,x,enc_features):
        for i in range (0,len(self.ch_list)-1):
            x= self.ml[i](x)
            enc_features_crp= self.crop(enc_features[i],x)
            x=torch.cat([x,enc_features_crp],dim=1)
            x= self.ml2[i](x)
        return x
            
    def crop(self, enc_features, x):
        _, _, H, W = x.shape
        enc_features   = torchvision.transforms.CenterCrop([H, W])(enc_features)
        return enc_features
    
    
    
    
##################################################
################### Verification #################

# model=segEncoder([3,64,128,256,512,1024])

# x= torch.randn(1,3,572,572)

# out=model(x)

# for i in out:
#     print(i.shape)


# model2 = segDecoder([1024, 512, 256, 128, 64])

# x2= torch.randn(1, 1024, 28, 28)

# output=model2(x2,out[::-1][1:])

# print(output.shape)
###################################################
    
    
    
    
import torch
import torch.nn as nn




class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv,self).__init__()
        self.dconv= nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,bias=False),  #Aladin added padding
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )
        
    def forward(self,x):
        out =self.dconv(x)
        return out
    
    
##################################################
################### Verification #################

# model=DoubleConv(3,64)

# x= torch.randn(1,3,572,572)

# out=model(x)

# print(out.shape)

###################################################


class segEncoder(nn.Module):
    def __init__(self,ch_list):
        super(segEncoder,self).__init__()
        self.ml=nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        for i in range(0,len(ch_list)-1):
            self.ml.append(DoubleConv(ch_list[i],ch_list[i+1]))
            
    def forward(self,x):
        features=[]
        for i in self.ml:
            x= i(x)
            features.append(x)
            x=self.pool(x)
        return features


##################################################
################### Verification #################

# model=segEncoder([3,64,128,256,512,1024])

# x= torch.randn(1,3,572,572)

# out=model(x)

# for i in out:
#     print(i.shape)

###################################################
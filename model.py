import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

torch.manual_seed(42)
    
    



class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self,norm_value = 'bn'):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,padding=1)
        self.norm1 = self.get_norm_layer(16,norm_value)
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3,padding=1)
        self.norm2 = self.get_norm_layer(16,norm_value)

        self.conv3 = nn.Conv2d(16,20,kernel_size =1,padding=0)

        self.conv4 = nn.Conv2d(16, 20, kernel_size=3,padding=1)
        self.norm4 = self.get_norm_layer(20,norm_value)

        self.conv5 = nn.Conv2d(20, 20, kernel_size=3,padding=1)
        self.norm5 = self.get_norm_layer(20,norm_value)

        self.conv6 = nn.Conv2d(20,20,kernel_size =3,padding=1)
        self.norm6 = self.get_norm_layer(20,norm_value)

        self.conv7 = nn.Conv2d(20,32,kernel_size =1,padding=0)
        self.norm7 = self.get_norm_layer(32,norm_value)

        self.conv8 = nn.Conv2d(32, 32, kernel_size=3,padding=1)
        self.norm8 = self.get_norm_layer(32,norm_value)

        self.conv9 = nn.Conv2d(32,48,kernel_size =3,padding=1)
        self.norm9 = self.get_norm_layer(48,norm_value)

        self.conv10 = nn.Conv2d(48,16,kernel_size =3)
        self.norm10 = self.get_norm_layer(16,norm_value)

        self.conv11 = nn.Conv2d(16, 10, 1, padding=0)
       

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x =  F.relu(self.norm2(x + self.conv2(x)))
        x = F.relu(F.max_pool2d(x, 2))
        x=   F.relu(self.norm4(self.conv4(x)))
        x=   F.relu(self.norm5(self.conv5(x)))
        x=   F.relu(self.norm6(self.conv6(x)))
        x=  F.relu(self.norm7(self.conv7(x)))
        x=  F.relu(F.max_pool2d(x ,2))
        x=  F.relu(self.norm8(self.conv8(x)))
        x=  F.relu(self.norm9(self.conv9(x)))
        x=  F.relu(self.norm10(self.conv10(x)))
        x=  nn.AvgPool2d(4)(x)
        x=  F.relu(self.conv11(x))
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)
    
    def get_norm_layer(self, num_channels, use_batch_norm):
      if use_batch_norm == 'bn':
        return nn.BatchNorm2d(num_channels)
      elif use_batch_norm == 'gn':
        return nn.GroupNorm(1, num_channels)
      else:
        return nn.GroupNorm(2,num_channels)
    

def model_summary(model,input_val):
    return summary(model, input_size=input_val)
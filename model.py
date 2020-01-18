from torch.utils.data import DataLoader,Dataset
import os
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# CNN for 2 layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.fc1 = nn.Linear(in_features = 14*14*64,out_features = 64)
        self.fc2 = nn.Linear(in_features = 64,out_features = 2)


    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1,self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        
        return num_features

 # CNN for 4 layers
class CNN_4L(nn.Module):
    def __init__(self):
        super(CNN_4L,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.fc1 = nn.Linear(in_features = 13*13*64,out_features = 128)
        self.fc2 = nn.Linear(in_features = 128,out_features = 64)
        self.fc_out = nn.Linear(in_features = 64,out_features = 2)

        self.drop_layer = nn.Dropout(p=0.2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = F.relu(self.conv3(x))
        x = self.drop_layer(F.max_pool2d(F.relu(self.conv4(x)),2))
        
        x = x.view(-1,13*13*64)

        x = self.drop_layer(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)

        return x
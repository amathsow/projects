from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from dataset import CatDogDataset
from model import CNN_4L
from model import CNN




image_size = (64, 64)
image_row_size = image_size[0] * image_size[1] * 3

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])



path    = '/home/aims/Downloads/cat-and-dog/'
train_data = CatDogDataset(path+'training_set/training_set/',  transform=transform)
test_data = CatDogDataset(path+'test_set/test_set/',  transform=transform)


net =  CNN()


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                         shuffle=False, num_workers=4)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("2 Layers:")

for epoch in range(10):

    r_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # gradient initialization
        optimizer.zero_grad()

        # compute forward loss backward and update gradient
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print results
        r_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, r_loss / 2000))
            r_loss = 0.0

print('Training Over for 2 Layers !!!')

#  python main.py -h
#    usage: main.py [-h] [-file_dir FILE_DIR] [-batch_size BATCH_SIZE] [-lr LR]
#           [-epoch EPOCH] [-model {SIMPLE,DEEPER}]


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images for 2 layers is: %d %%' % (100 * correct / total))





net_4L =  CNN_4L()


train1_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True, num_workers=4)
test1_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                         shuffle=False, num_workers=4)


criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net_4L.parameters(), lr=0.001, momentum=0.9)

print("4 Layers:")

for epoch in range(10):

    r1_loss = 0.0
    for i, data in enumerate(train1_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # gradient initialization
        optimizer1.zero_grad()

        # compute forward loss backward and update gradient
        outputs = net_4L(inputs)
        loss1 = criterion1(outputs, labels)
        loss1.backward()
        optimizer1.step()

        # print results
        r1_loss += loss1.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, r_loss / 2000))
            r1_loss = 0.0

print('Training Over for 4 Layers !!!')

#  python main.py -h
#    usage: main.py [-h] [-file_dir FILE_DIR] [-batch_size BATCH_SIZE] [-lr LR]
#           [-epoch EPOCH] [-model {SIMPLE,DEEPER}]


correct1 = 0
total1 = 0
with torch.no_grad():
    for data in test1_loader:
        images, labels = data
        outputs = net_4L(images)
        _, predicted1 = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted1 == labels).sum().item()

print('Accuracy on test images for 4 layers is: %d %%' % (100 * correct1 / total1))


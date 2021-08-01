# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 23:07:37 2021

@author: Jericho
"""
from split_data import Leaves_Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# %%

# training & validation dataloader

training_data = pd.read_csv (r'train.csv')
testing_data = pd.read_csv (r'test.csv')

train, val = train_test_split(training_data, test_size=0.15)

resize_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

datapath = 'images'
train_data = Leaves_Dataset(csv_file = train, root_dir = datapath, transform = resize_transform)
val_data = Leaves_Dataset(csv_file = val, root_dir = datapath, transform = resize_transform)

BATCH_SIZE = 128

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=False)


for step, (image, label) in enumerate(train_loader):
    print("label: ", label.shape)
    print("image: ", image.shape)
    break

for step, (image, label) in enumerate(valid_loader):
    print("label: ", label.shape)
    print("image: ", image.shape)
    break


# %%
# Hyperparameters
RANDOM_SEED = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

NUM_CLASSES = 176 # FOR VIOLENCE DETECTION (BIONARY CIASSIFICATION PROBLEM)
DEVICE = 'cuda:0' 

# %%

# define basic blocks (or the "bottleneck")

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(block, self).__init__()
        
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = x + identity # Skip connection
        x = self.relu(x)
        
        return x
            


# In[11]:


class ResNet(nn.Module): # layers (a list) is used to make different type of ResNet
    def __init__(self, block, layers, image_channels, num_classes): # image_channels: 1 for grayscale, 3 for colored image
        super(ResNet, self).__init__()
        
        # this is just the first layer, this layer does not have any residual block
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # ResNet layers (4 blocks as shown in the paper)
        self.layer1 = self.make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self.make_layer(block, layers[1], out_channels = 128, stride = 2)
        self.layer3 = self.make_layer(block, layers[2], out_channels = 256, stride = 2)
        self.layer4 = self.make_layer(block, layers[3], out_channels = 512, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)

        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
        

        
    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1, 
                                                          stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) # 256 -> 64 -> 64*4 (256) again
            
        return nn.Sequential(*layers)
        


# In[12]:


def ResNet50(img_channels, num_classes):
    return ResNet(block, [3,4,6,3], 3, NUM_CLASSES)

model = ResNet50(3, NUM_CLASSES)


# %%
# # use pre-trained model on imagenet
# model = models.resnet50(pretrained = True)
# # freeze some weights
# for param in model.parameters():
#     param.requires_grad = False
# # # layer4 is unfrozen, can be updated
# model.layer4.requires_grad = True
# # fc layer is modified so that the output has 176 classes
# model.fc = nn.Linear(2048, 176)


# input_tensor = torch.autograd.Variable(torch.rand(16, 3, 224, 224))
# # model = model(class_num=NUM_CLASSES)
# output = model(input_tensor)
# print(input_tensor.shape)
# print(output.shape)

# %%
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%

def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    

# minibatch_cost, epoch_cost = [], []
# all_train_acc, all_valid_acc = [], []

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        # minibatch_cost.append(cost)
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        train_acc = compute_accuracy(model, train_loader)
        valid_acc = compute_accuracy(model, valid_loader)
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, NUM_EPOCHS, train_acc, valid_acc))
        
        # writer.add_scalars('Accuracy', {'training': train_acc, 
                                        # 'validation': valid_acc},epoch)
        
        # all_train_acc.append(train_acc)
        # all_valid_acc.append(valid_acc)
        cost = compute_epoch_loss(model, train_loader)
        
        # writer.add_scalar('Loss', cost, epoch)
        # epoch_cost.append(cost)
        

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    # if not epoch % 5:
        # writer.flush()
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))







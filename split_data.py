# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:50:00 2021
MSSC 5931 CLASS PROJECT 
JERRY SU
@author: Jericho (Kaggle ID)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import os
import torch
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


training_data = pd.read_csv (r'train.csv')
testing_data = pd.read_csv (r'test.csv')



leaves_labels = sorted(list(set(training_data['label'])))
n_classes = len(leaves_labels)

# key is class name
class_to_num = dict(zip(leaves_labels, range(n_classes)))

# key is number
num_to_class = {v : k for k, v in class_to_num.items()}


train, val = train_test_split(training_data, test_size=0.15)


class Leaves_Dataset(Dataset):

    def __init__(self, csv_file, root_dir, test = None, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.leaves_frame = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.test = test
        
        
        

        # # list comprehension
        # status_dict = sorted(self.leaves_frame['label'].unique().tolist())
        # # create a mapping (dictionary) between label and index
        # self.label2index = {label:index for index, label in enumerate(status_dict)} 
        
        # # save the dictionary 
        # dirs = 'dictionary.txt'
        # if not os.path.exists(dirs):
        #     with open('dictionary.txt', 'w') as f:
        #         for key in self.label2index.keys():
        #             f.write("%s %s\n"%(key, self.label2index[key]+1))
        
                    
        # # read the .txt as a dictionary, read numerical label from this dictionary 
        # self.d = {}
        # with open("dictionary.txt") as f:
        #     for line in f:
        #         (key, val) = line.split()
        #         # print(key)
        #         self.d[key] = val
                

    def __len__(self):
        return len(self.leaves_frame)
    
    def __getitem__(self, idx):
        image_path = self.leaves_frame.iloc[idx, 0]
        image = io.imread(image_path)
        # convert (height, width, channel) to (channel, height, width)
        # image = np.transpose(image, (2,0,1))
        # print(image.shape)
        
        
        if self.test:
            
            if self.transform:
                image = self.transform(image)
            
            return image
        
        else:
            
            image_name = self.leaves_frame.iloc[idx, 1]
            
            # print(image_name)
            
            # get the numerical label from the dictionary d
            # num_label = self.d[image_name]
            # print(self.d[image_name])
            
            # print(num_label.type)
            num_label = class_to_num[image_name]
            
            # needs to be converted to int type here
            y_label = torch.tensor(int(num_label))
            
            if self.transform:
                image = self.transform(image)
    
            return (image, y_label)

# %%
if __name__ == '__main__':
    
    
    resize_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    
    resize_transform_valid = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    
    
    BATCH_SIZE = 1024
    
    datapath = 'images'
    
    train_data = Leaves_Dataset(csv_file = train, root_dir = datapath, transform = resize_transform)
    val_data = Leaves_Dataset(csv_file = val, root_dir = datapath, transform = resize_transform_valid)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    
    
    test_data = Leaves_Dataset(csv_file = testing_data, test = True, root_dir = datapath, transform = resize_transform_valid)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    
    
    
    for step, (image, label) in enumerate(train_loader):
        print("label: ", label.shape)
        print("image: ", image.shape)
        break
    
    for step, (image, label) in enumerate(val_loader):
        print("label: ", label.shape)
        print("image: ", image.shape)
        break
    
    for step, image in enumerate(test_loader):
        print("image: ", image.shape)
        break
    
# %%
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrain model & modify it
    # model = torchvision.models.inception_v3(pretrained=True)
    model = torchvision.models.resnet50(pretrained=True)
    
    # transfer learning
    
    for param in model.parameters():
                param.requires_grad = False
    
                
    model.layer2.requires_grad = True
    model.layer3.requires_grad = True
    model.layer4.requires_grad = True
    model.fc = nn.Linear(2048, 176)
    
    model.to(device)
    
# %%    
    # Hyperparameters
    # num_classes = 10
    learning_rate = 1e-3
    # batch_size = 16
    num_epochs = 30
    # weight_decay= 1e-3
    model_path = './pre_res_model.ckpt'
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    # Train Network
    for epoch in range(num_epochs):
        losses = []
        train_acc = []
        
        # for batch_idx, (data, targets) in enumerate(train_loader):
        for batch_idx in tqdm(train_loader):
            data, targets = batch_idx
            
            # print(targets.shape)
            # print(data.shape)
            
            
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            
    
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
    
            losses.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
    
            # gradient descent or adam step
            optimizer.step()
            
            # print(scores.argmax(dim = -1))
            # print(targets)
            # print((scores.argmax(dim=-1) == targets).float())
            
            acc = (scores.argmax(dim=-1) == targets).float().mean()
            # print(acc)
            
            train_acc.append(acc)
            
            
        train_acc = sum(train_acc) / len(train_acc)
        train_cost =  sum(losses)/len(losses)
            
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] Cost = {train_cost:.5f}, Train_acc = {train_acc:.5f}")
        
        # print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")
        # print(f"Train ACC at epoch {epoch} is {train_acc:.5f}")
        
        
                # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []
        
        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):
            imgs, labels = batch
            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))
                
            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))
    
            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
    
            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            
        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
    
        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, Val_acc = {valid_acc:.5f}")
        
        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))

    
# %%
    
    saveFileName = './submission.csv'
    
    
    # test_data = Leaves_Dataset(csv_file = testing_data, root_dir = datapath, transform = resize_transform_valid)
    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
    
    
    ## predict
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 176)
    
    
    # create model and load weights from checkpoint
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()
    
    # Initialize a list to store the predictions.
    predictions = []
    # Iterate the testing set by batches.
    for batch in tqdm(test_loader):
        
        imgs = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        
        # Take the class with greatest logit as prediction and record it.
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    
    preds = []
    test_path = 'test.csv'
    for i in predictions:
        preds.append(num_to_class[i])
    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    
    
    




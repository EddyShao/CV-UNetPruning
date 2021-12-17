import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from models.unet import UNet
from data.drone import DroneDataset, DroneTestDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda")

from metric.unetMetrics import pixel_accuracy, MiOU
from UnetTrainer import fit



IMAGE_PATH = '/scratch/zs1542/drone_dataset/new_size_img/'
MASK_PATH = '/scratch/zs1542/drone_dataset/new_size_mask/'

save_dir = "/scratch/zs1542/CV-FinalProject/CVFramework/checkpoints/"



# create df with id of the dataset
def create_df(path):
    name = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': name}, index = np.arange(0, len(name)))

df = create_df(IMAGE_PATH)
# print('Total Images: ', len(df))

#split the dataset into train, validation and test data
X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=0)
X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)

 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


#create datasets
train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
#load data
batch_size= 3
n_class = 24

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

print("Done with data.\n")
print("Start training...")

model = UNet(out_classes=24).to(device)

lr = 0.001
epoch = 30
weight_decay = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epoch,
                                            steps_per_epoch=len(train_loader))


train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy = fit(
    epoch, model, train_loader, val_loader, criterion, optimizer, scheduler, batch_size, n_class, device, save_dir
)











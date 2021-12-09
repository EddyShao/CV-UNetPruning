import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torchvision import transforms as T
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# change your own image_path and mask_path
IMAGE_PATH = '/scratch/qz1086/drone_dataset/images/'
MASK_PATH = '/scratch/qz1086/drone_dataset/dataset/label_processed/'

# Drone Dataset:
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]


class DroneDataset(Dataset):

    def __init__(self, 
                 img_path, 
                 mask_path, 
                 X, 
                 mean, 
                 std, 
                 mode='train',
                 transform=None):

        self.mode = mode
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std


    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('paved-area', (128, 64, 128)),
            ('dirt', (130, 76, 0)),
            ('grass', (0, 102, 0)),
            ('gravel', (112,  103,  87)),
            ('water', (28,	42,	168)),
            ('rocks', (48,	41,	30)),
            ('pool', (0,  50,  89)),
            ('vegetation', (107, 142, 35)),
            ('roof', (70, 70, 70)),
            ('wall', (102,102, 156)),
            ('window', (254,  228,	12)),
            ('door', (254,	148,  12)),
            ('fence', (190,	153, 153)),
            ('fence-pole', (153, 153, 153)),
            ('person', (255, 22, 96)),
            ('dog', (102, 51, 0)),
            ('car', (9,	143, 150)),
            ('bicycle', (119, 11, 32)),
            ('tree', (51, 51, 0)),
            ('bald-tree', (190, 250, 190)),
            ('ar-marker', (112, 150, 146)),
            ('obstacle', (2, 135, 115)),
            ('conflicting', (255, 0, 0))
    ])  


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = cv2.imread(self.img_path + self.X[i] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + "{:03d}".format(int(self.X[i])) + '.png', cv2.IMREAD_GRAYSCALE)

        # how to transform?
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = Image.fromarray(augmented['image'])
            mask = augmented['mask']
        else:
            img = Image.fromarray(img)
        trans = T.Compose(
            [T.ToTensor(),
             T.Normalize(self.mean, self.std)]
        )
        img = trans(img)
        # simply calling the below will not work, you have to forward it to return something,
        # thus, we have to use T.Compose and then call it.
        # img = T.ToTensor()
        # img = T.Normalize(self.mean, self.std)
        mask = torch.from_numpy(mask).long()

        # if self.patches:
        #     img, mask = self.tiles(img, mask)

        return img, mask



if __name__ == "__main__":
    # Demo for loading train dataset
    # create df with id of the dataset
    def create_df(path):
        name = []
        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                name.append(filename.split('.')[0])

        return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


    df = create_df(IMAGE_PATH)
    print('Total Images: ', len(df))


    # split the dataset into train, validation and test data
    X_trainval, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=0)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=0)


    # Define mean and std value
    # Drone Dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # create datasets
    train_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_train, mean, std)
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std)
    # load data--->define batch size
    batch_size = 32

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True)
    val_loader = DataLoader(val_set, 
                            batch_size=batch_size, 
                            shuffle=True)

    
    for i, batch in enumerate(dataloader):
        print(batch)
        break
    


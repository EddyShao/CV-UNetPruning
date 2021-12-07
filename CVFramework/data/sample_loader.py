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

# import two dataset class
from dataset import DroneDataset, DroneTestDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = 'dataset/img/'
MASK_PATH = 'dataset/mask/'


# create df with id of the dataset
def create_df(path):
    name = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))


df = create_df(IMAGE_PATH)
# print('Total Images: ', len(df))


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
batch_size = 3

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
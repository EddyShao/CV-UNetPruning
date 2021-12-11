import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# Drone Dataset basic info:
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
height, width = 200, 300

class_encoding = OrderedDict([
            ("unlabeled", (0, 0, 0)),
            ("paved-area", (28, 64, 128)),
            ("dirt", (130, 76, 0)),
            ("grass", (0, 102, 0)),
            ("gravel", (112, 103, 87)),
            ("water", (28, 42, 168)),
            ("rocks", (48, 41, 30)),
            ("pool", (0, 50, 89)),
            ("vegetation", (107, 142, 35)),
            ("roof", (70, 70, 70)),
            ("wall", (102, 102, 156)),
            ("window", (254, 228, 12)),
            ("door", (254, 148, 12)),
            ("fence", (190, 153, 153)),
            ("fence-pole", (153, 153, 153)),
            ("person", (255, 22, 96)),
            ("dog", (102, 51, 0)),
            ("car", (9, 143, 150)),
            ("bicycle", (119, 11, 32)),
            ("tree", (51, 51, 0)),
            ("bald-tree", (190, 250, 190)),
            ("ar-marker", (112, 150, 146)),
            ("obstacle", (2, 135, 115)),
            ("conflicting", (255, 0, 0)),
    ])  


class DroneDataset(Dataset):

    def __init__(self, img_path, mask_path, X, mean, std, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.mean = mean
        self.std = std
        self.class_encoding = class_encoding
        

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
            [T.ToTensor(), T.Resize((height, width)),
             T.Normalize(self.mean, self.std)]
        )
        img = trans(img)
        # simply calling the below will not work, you have to forward it to return something,
        # thus, we have to use T.Compose and then call it.
        # img = T.ToTensor()
        # img = T.Normalize(self.mean, self.std)
        mask = torch.from_numpy(mask).long()
        mask = mask.view(1, 4000, 6000)

        label_trans = T.Compose(
            [T.Resize((height, width), Image.NEAREST)]
        )
        mask = label_trans(mask)
        # if self.patches:
        #     img, mask = self.tiles(img, mask)

        mask = mask.view(200, 300)

        return img, mask


class DroneTestDataset(Dataset):

    def __init__(self, img_path, mask_path, X):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + "{:03d}".format(int(self.X[idx])) + '.png', cv2.IMREAD_GRAYSCALE)

        mask = torch.from_numpy(mask).long()

        trans = T.Compose(
            [T.ToTensor(), T.Resize((height, width)),
             T.Normalize(self.mean, self.std)]
        )
        img = trans(img)
        # simply calling the below will not work, you have to forward it to return something,
        # thus, we have to use T.Compose and then call it.
        # img = T.ToTensor()
        # img = T.Normalize(self.mean, self.std)
        mask = torch.from_numpy(mask).long()
        mask = mask.view(1, 4000, 6000)

        label_trans = T.Compose(
            [T.Resize((height, width), Image.NEAREST)]
        )
        mask = label_trans(mask)
        # if self.patches:
        #     img, mask = self.tiles(img, mask)

        mask = mask.view(200, 300)

        return img, mask





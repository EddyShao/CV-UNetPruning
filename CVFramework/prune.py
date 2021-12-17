from math import e
import os
from re import I
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

import torch.nn.utils.prune as prune

# import tensorflow as tf
# import segmentation_models_pytorch as smp

# for image augmentation
# import albumentations as A

from sklearn.model_selection import train_test_split

from PIL import Image
import cv2

from models.unet import UNet
from data.drone import DroneDataset, DroneTestDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from metric.unetMetrics import pixel_accuracy, MiOU

from UnetTrainer import fit
# from thop import profile

device = torch.device("cuda")
# device = torch.device("cuda")



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

# train_loader = DataLoader(train_set)
# val_loader = DataLoader(val_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)



#############################################################################################
state_dict_path = "/scratch/zs1542/CV-FinalProject/CVFramework/checkpoints_1211_0/Unet_28.pth"
print("We are using the state dict from\n", state_dict_path)
#############################################################################################




criterion = nn.CrossEntropyLoss()
print("Prune the ENCODER only...")
print("Encoder pruning\n amount: 0 - 1 \nmethod: Global Unstructured L1")




encoder_out = {}
for amount in np.linspace(0, 1, 21):
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    parameters_to_prune = (
    (model.double_conv1[0], 'weight'),
    (model.double_conv1[2], 'weight'),

    (model.double_conv2[0], 'weight'),
    (model.double_conv2[2], 'weight'),

    (model.double_conv3[0], 'weight'),
    (model.double_conv3[2], 'weight'),

    (model.double_conv4[0], 'weight'),
    (model.double_conv4[2], 'weight'),
    )
    
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.RandomUnstructured,
    amount=amount,
    )

    print("AMOUNT: ", amount)

    for module, dummy in parameters_to_prune:
        #prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

        total_loss = 0
        total_acc = 0
        total_m = 0

    start = time.time()

    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        acc_i = pixel_accuracy(pred, mask)
        m_i = MiOU(pred, mask, n_classes=24)
        total_loss += loss.item()
        total_m += m_i
        total_acc += float(acc_i)
    
    end = time.time()

    duration = end - start

    loss, acc, m = total_loss/len(val_loader), total_acc/len(val_loader), total_m/ len(val_loader)
    
    print("loss: ", loss)
    print("acc: ", acc)
    print("MIOU: ", m)
    print("Duration: ", duration)

    encoder_out[amount] = (loss, acc, m, duration)

    ### Save model for visualziation
    model_file = save_dir + "encoder" + "_" + str(int(amount*100)) + '.pth'
    torch.save(model.state_dict(), model_file, _use_new_zipfile_serialization=True)
    print('\nSaved model to ' + model_file + '.'+"\n")




print(encoder_out)

#==================================================================================================#

print("Prune the DECODER only...")



decoder_out = {}
for amount in np.linspace(0, 1, 21):
    # Initialize the modoel 
    model = UNet(out_classes=24).to(device)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    parameters_to_prune = (
    (model.up_conv1, 'weight'),
    (model.up_conv2, 'weight'),
    (model.up_conv3, 'weight'),
    (model.up_conv4, 'weight'),

    (model.up_double_conv1[0], "weight"),
    (model.up_double_conv1[2], "weight"),

    (model.up_double_conv2[0], "weight"),
    (model.up_double_conv2[2], "weight"),

    (model.up_double_conv3[0], "weight"),
    (model.up_double_conv3[2], "weight"),

    (model.up_double_conv4[0], "weight"),
    (model.up_double_conv4[2], "weight"),
    )
    
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.RandomUnstructured,
    amount=amount,
    )

    for module, dummy in parameters_to_prune:
        prune.remove(module, 'weight')
        print(
        "Density in " + str(module)+ " : {:.2f}%".format(
            100. * float(torch.sum(module.weight != 0))
            / float(module.weight.nelement())
        )
        )

    total_loss = 0
    total_acc = 0
    total_m = 0

    start = time.time()
    
    for i, (img, mask) in enumerate(val_loader):
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        acc_i = float(pixel_accuracy(pred, mask))
        m_i = MiOU(pred, mask, n_classes=24)
        total_loss += loss.item()
        total_m += m_i
        total_acc += float(acc_i)

    end = time.time()
    
    duration = end - start

    loss, acc, m = total_loss/len(val_loader), total_acc/len(val_loader), total_m/ len(val_loader)
    print("loss: ", loss)
    print("acc: ", acc)
    print("MIOU: ", m)
    print("Duration: ", duration)

    decoder_out[amount] = (loss, acc, m, duration)


    ### Save model for visualziation
    model_file = save_dir + "decoder" + "_" + str(int(amount*100)) + '.pth'
    torch.save(model.state_dict(), model_file, _use_new_zipfile_serialization=True)
    print('\nSaved model to ' + model_file + '.'+"\n")


    
print(decoder_out)




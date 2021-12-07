import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

import transforms as ext_transforms
from models.enet import ENet
from metric.iou import IoU
import utils
from data.cityscapes import Cityscapes

from train import Train
from test import Test

dataset_args = {
    "dataset": "cityscapes",
    "dataset_dir": "/scratch/qz1086/cityscapes",
    "save_dir": "",
    "height": 320,
    "width": 480,
    "batch_size": 32,
    "workers":0,
}

def load_dataset(dataset): 
    # Here this parameter should be one of the dataloader defined in directory "data"
    print("\nLoading dataset...\n")

    print("Selected dataset:", dataset_args["dataset"])
    print("Dataset directory:", dataset_args["dataset_dir"])
    print("Save directory:", dataset_args["save_dir"])

    image_transform = transforms.Compose(
        [transforms.Resize((dataset_args["height"], dataset_args["width"])),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((dataset_args["height"], dataset_args["width"]), Image.NEAREST),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        root_dir=dataset_args["dataset_dir"],
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=dataset_args["batch_size"],
        shuffle=True,
        num_workers=dataset_args["workers"])

    # Load the validation set as tensors
    val_set = dataset(
        root_dir=dataset_args["dataset_dir"],
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    
    val_loader = data.DataLoader(
        val_set,
        batch_size=dataset_args["batch_size"],
        shuffle=False,
        num_workers=dataset_args["workers"])

    # Load the test set as tensors
    test_set = dataset(
        dataset_args["dataset_dir"],
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=dataset_args["batch_size"],
        shuffle=False,
        num_workers=dataset_args["workers"])

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding


    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    return (train_loader, val_loader, test_loader), class_encoding


train_args = {
    "model": ENet,
    "learning_rate": 0.01,
    "resume": False,
    "save_dir": "/scratch/qz1086/CV-FinalProject/CVFramework/checkpoints/",
    "epochs": 60,
    "report_step":10,
    "save_filename": "ENet"

}

def train(train_loader, val_loader, class_encoding, device=None):
    print("\nTraining...\n")
    device = torch.device("cuda")

    num_classes = len(class_encoding)

    # Intialize model
    model = train_args["model"](num_classes).to(device)
    # Check if the network architecture is correct
    # Omit this process for now, as the model is too bigggggg
    # print(model)

    criterion = nn.CrossEntropyLoss()

    # ENet authors used Adam as the optimizer
    # However, use SGD for simplicity here
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_args["learning_rate"],
        )

    # Learning rate decay scheduler
    # lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
    #                                  args.lr_decay)

    # Evaluation metric
    metric = IoU(num_classes)

    # Optionally resume from a checkpoint

    start_epoch = 0
    best_miou = 0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, train_args["epochs"]):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        epoch_loss, (iou, miou) = train.run_epoch(False)
        # lr_updater.step()

        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        if (epoch + 1) % train_args["report_step"] == 0 or epoch + 1 == train_args["epochs"]:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(True)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == train_args["epochs"] or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            if miou > best_miou:
                print("\nBest model thus far. \n")
                best_miou = miou
                print("Best miou", best_miou)

        model_file = train_args["save_dir"] + train_args["save_filename"] + "_" + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '.'+"\n")

    return model



(train_loader, val_loader, test_loader), class_encoding = load_dataset(Cityscapes)


train(train_loader, val_loader, class_encoding)
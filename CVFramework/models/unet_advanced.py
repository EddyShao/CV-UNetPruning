# https://github.com/qubvel/segmentation_models.pytorch
import torch
import segmentation_models_pytorch as smp
import time
import numpy as np
import pandas as pd

model = smp.Unet(
    'mobilenet_v2',
    encoder_weights='imagenet',
    classes=19,
    encoder_depth=5,
    decoder_channels=[256, 128, 64, 32, 16]
    )
#Evaluation Matrices
def pixel_accuracy(output,label):
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    accur = torch.eq(output, label).int()
    return (torch.sum(accur).float() / output.nelement())

def MiOU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

# Training-fit process
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, batch_size, n_class=23):
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    train_miou = []
    val_miou = []
    train_accuracy = []
    val_accuracy = []
    mini_loss = float('inf')
    no_progress = 0

    model.to(device)
    train_begin = time.time()

    for epoch in range(epochs):

        begin_timer = time.time()
        total_loss = 0
        total_miou = 0
        total_accuracy = 0

        # start training
        model.train()
        for i, batch in enumerate(train_loader):  # get batch
            img, mask = batch
            # !!tensor and model are different, not inplace
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img)  # pass batch
            loss = criterion(prediction, mask)  # calculate loss, loss tensor
            # add to evaluation metrics
            total_miou += MiOU(prediction, mask)
            total_accuracy += pixel_accuracy(prediction, mask)

            # back prop
            optimizer.zero_grad()  # when processing a new batch, clear the gradient on start
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights

            # TBD: update learning rate...
            scheduler.step()

            total_loss += loss.item()

        # validation
        else:
            model.eval()
            total_val_loss = 0
            total_val_miou = 0
            total_val_accuracy = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    img, mask = batch
                    # !!tensor and model are different, not inplace
                    img = img.to(device)
                    mask = mask.to(device)

                    prediction = model(img)
                    loss = criterion(prediction, mask)  # how to split mask from the train_loader
                    total_val_loss += loss.item()

                    total_val_miou += MiOU(prediction, mask, n_class)
                    total_val_accuracy += pixel_accuracy(prediction, mask)

                    total_val_loss += loss.item()

                # calculate loss
                this_train_loss = total_loss / len(train_loader)
                this_val_loss = total_val_loss / len(val_loader)

                train_losses.append(this_train_loss)
                val_losses.append(this_val_loss)

                ## determine when to stop the train through the epochs
                if mini_loss > this_val_loss:
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(mini_loss, this_val_loss))
                    mini_loss = this_val_loss
                else:
                    no_progress += 1
                    mini_loss = this_val_loss
                    print(f'Loss Not Decrease for {no_progress} time')
                    if no_progress == 10:
                        print('Loss not decrease for 10 times, Stop Training')
                        break

                # calculate iou
                this_miou = total_miou / len(train_loader)
                this_val_miou = total_val_miou / len(val_loader)

                train_miou.append(this_miou)
                val_miou.append(this_val_miou)

                # calculate accuracy
                this_accuracy = total_accuracy / len(train_loader)
                this_val_accuracy = total_val_accuracy / len(val_loader)
                train_accuracy.append(this_accuracy)
                val_accuracy.append(this_val_accuracy)

                print("Epoch:{}/{}..".format(epoch + 1, epochs),
                      "Train Loss: {:.3f}..".format(this_train_loss),
                      "Val Loss: {:.3f}..".format(this_val_loss),
                      "Train mIoU:{:.3f}..".format(this_miou),
                      "Val mIoU: {:.3f}..".format(this_val_miou),
                      "Train Acc:{:.3f}..".format(this_accuracy),
                      "Val Acc:{:.3f}..".format(this_val_accuracy),
                      "Time: {:.2f}m".format((time.time() - begin_timer) / 60))

    duration = time.time() - train_begin
    print('Total time: {:.2f} m'.format(duration / 60))

    return train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy
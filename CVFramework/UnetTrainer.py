import torch
import time
import numpy as np
import pandas as pd
import torch.functional as F
from metric.unetMetrics import pixel_accuracy, MiOU



def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler,  batch_size, n_class, device, save_dir):
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
        for i, batch in enumerate(train_loader): # get batch
            img, mask = batch
            # !!tensor and model are different, not inplace 
            img = img.to(device)
            mask = mask.to(device)

            prediction = model(img) # pass batch
            loss = criterion(prediction,mask) # calculate loss, loss tensor
            # add to evaluation metrics
            total_miou += MiOU(prediction,mask) 
            total_accuracy += pixel_accuracy(prediction,mask)

            # back prop
            optimizer.zero_grad() # when processing a new batch, clear the gradient on start
            loss.backward() # calculate gradients
            optimizer.step() # update weights

            #TBD: update learning rate...
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

                    total_val_miou += MiOU(prediction,mask,n_class)
                    total_val_accuracy += pixel_accuracy(prediction, mask)

                    total_val_loss += loss.item()



                # calculate loss
                this_train_loss = total_loss/len(train_loader)
                this_val_loss = total_val_loss/len(val_loader)

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
                this_miou = total_miou/len(train_loader)
                this_val_miou = total_val_miou/len(val_loader)

                train_miou.append(this_miou)                
                val_miou.append(this_val_miou)

                # calculate accuracy
                this_accuracy = total_accuracy/len(train_loader)
                this_val_accuracy = total_val_accuracy/ len(val_loader)
                train_accuracy.append(this_accuracy)
                val_accuracy.append(this_val_accuracy)

                print("Epoch:{}/{}..".format(epoch+1, epochs),
                    "Train Loss: {:.3f}..".format(this_train_loss),
                    "Val Loss: {:.3f}..".format(this_val_loss),
                    "Train mIoU:{:.3f}..".format(this_miou),
                    "Val mIoU: {:.3f}..".format(this_val_miou),
                    "Train Acc:{:.3f}..".format(this_accuracy),
                    "Val Acc:{:.3f}..".format(this_val_accuracy),
                    "Time: {:.2f}m".format((time.time()-begin_timer)/60))

                model_file = save_dir + "Unet_" + str(epoch) + '.pth'
                torch.save(model.state_dict(), model_file)
                print('\nSaved model to ' + model_file + '.'+"\n")

    duration = time.time() - train_begin
    print('Total time: {:.2f} m' .format(duration/60))

    return train_losses, val_losses, train_miou, val_miou, train_accuracy, val_accuracy
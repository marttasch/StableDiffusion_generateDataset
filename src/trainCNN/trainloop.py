import torch
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision
from torch.utils.data import DataLoader
import numpy as np
import datetime

# --- Function for Trainloop ---
def trainloop(model, config, device, class_names, train_set, test_set, tensorboard, output_folder):
    model = model.to(device)

    # --- init ---
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    tensorboard.init_tensorboard()

    # --- Metrics ---
    train_accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)
    train_recall = MulticlassRecall(num_classes=len(class_names)).to(device)
    train_precision = MulticlassPrecision(num_classes=len(class_names)).to(device)
    test_accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)
    test_recall = MulticlassRecall(num_classes=len(class_names)).to(device)
    test_precision = MulticlassPrecision(num_classes=len(class_names)).to(device)

    train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=True)

    # --- Training Loop ---
    startTime = datetime.datetime.now()
    for epoch in range(config['epochs']):

        epoch_loss_train, epoch_loss_test = [], []    # store loss for each epoch

        # --- Training ---
        for imgs, labels in train_loader:
            model.train()   # setup model for training
            imgs, labels = imgs.to(device), labels.to(device)   # move data to device

            # -- forward pass --
            preds = model(imgs)               # get predictions
            loss_train = criterion(preds, labels)   # calculate loss

            # -- backward pass --
            optimizer.zero_grad()   # reset gradients
            loss_train.backward()   # calculate gradients
            optimizer.step()       # update weights

            # -- update metrics --
            t_a = train_accuracy(preds, labels)
            #print(t_a)
            train_recall(preds, labels)
            train_precision(preds, labels)

            epoch_loss_train.append(loss_train.item())   # store loss


        # --- Testing ---
        with torch.no_grad():   # no need to track gradients for testing
            for imgs, labels in test_loader:   # iterate over test data
                model.eval()   # set model to evaluation mode
                imgs, labels = imgs.to(device), labels.to(device)   # move data to device

                # -- forward pass --
                preds = model(imgs)
                loss_test = criterion(preds, labels)

                # -- update metrics --
                test_accuracy(preds, labels)
                test_recall(preds, labels)
                test_precision(preds, labels)

                epoch_loss_test.append(loss_test.item())   # store loss


        # --- Epoch Summary ---
        # -- calculate metrics --
        train_acc = train_accuracy.compute()
        train_rec = train_recall.compute()
        train_pre = train_precision.compute()
        test_acc = test_accuracy.compute()
        test_rec = test_recall.compute()
        test_pre = test_precision.compute()

        #if epoch % 4 == 0:
        #  torch.save(model.state_dict(), "./model.pth")

        if epoch % config['safe_model_intervall'] == 0:
            torch.save(model.state_dict(), output_folder + f"/model_{epoch}.pth")

        train_accuracy.reset()
        train_recall.reset()
        train_precision.reset()
        test_accuracy.reset()
        test_recall.reset()
        test_precision.reset()

        # -- alculate average loss --
        epoch_loss_train = np.mean(epoch_loss_train)
        epoch_loss_test = np.mean(epoch_loss_test)

        # Write Tensor Board
        tensorboard.write_board(epoch, epoch_loss_train, train_acc, train_rec, train_pre, epoch_loss_test, test_acc, test_rec, test_pre)

        # -- print epoch summary --
        print(f"Epoch {str(epoch).zfill(3)}")
        print(f"TRAIN\t loss: {epoch_loss_train:.4f} \t acc: {train_acc * 100:.4f}")
        print(f"TEST\t loss: {epoch_loss_test:.4f} \t acc: {test_acc * 100:.4f}")
        print()

        # --- Update lr sheduler ---
        lr_scheduler.step()

    tensorboard.writer.flush()
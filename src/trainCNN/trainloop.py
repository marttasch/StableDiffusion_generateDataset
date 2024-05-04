import torch
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
import datetime
import os
import json

# --- Function for Trainloop ---
def trainloop(model, config, device, class_names, train_set, test_set, output_folder, mean, std, tensorboard  , logging):
    model = model.to(device)

    # --- init ---
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = torch.nn.CrossEntropyLoss()
    tensorboard.init_tensorboard()

    # --- Log ---
    log = {}
    log['config'] = config
    log['class_names'] = class_names
    log['training_set'] = [str(train_set)]
    log['test_set'] = [str(test_set)]
    log['mean'] = [str(mean)]
    log['std'] = [str(std)]
    log['device'] = str(device)
    log['model'] = model.__class__.__name__
    log['optimizer'] = optimizer.__class__.__name__
    log['lr_scheduler'] = {
        'name': lr_scheduler.__class__.__name__,
        'step_size': (lr_scheduler.step_size if hasattr(lr_scheduler, 'step_size') else None),  # StepLR
        'gamma': (lr_scheduler.gamma if hasattr(lr_scheduler, 'gamma') else None),  # StepLR
        'factor': (lr_scheduler.factor if hasattr(lr_scheduler, 'factor') else None),  # ReduceLROnPlateau
        'patience': (lr_scheduler.patience if hasattr(lr_scheduler, 'patience') else None),  # ReduceLROnPlateau
        'mode': (lr_scheduler.mode if hasattr(lr_scheduler, 'mode') else None),  # ReduceLROnPlateau
    }
    log['criterion'] = criterion.__class__.__name__

    # --- Write Log ---
    with open(os.path.join(output_folder, 'log.json'), 'w') as f:
        json.dump(log, f, indent=4)


    # --- Metrics ---
    train_accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)
    train_recall = MulticlassRecall(num_classes=len(class_names)).to(device)
    train_precision = MulticlassPrecision(num_classes=len(class_names)).to(device)
    test_accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)
    test_recall = MulticlassRecall(num_classes=len(class_names)).to(device)
    test_precision = MulticlassPrecision(num_classes=len(class_names)).to(device)

    train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=config['batch_size'], shuffle=True)

    # --- early stopping ---
    best_test_loss = float('inf')  # Initialize best validation loss
    patience = config['patience']  # Patience parameter for early stopping
    early_stopping_counter = 0  # Initialize early stopping counter

    # --- Training Loop ---
    startTime = datetime.datetime.now()
    for epoch in range(config['epochs']):
        start_time = datetime.datetime.now()  # Record start time of epoch
        
        # Logging
        epoch_loss_train, epoch_loss_test = [], []    # store loss for each epoch
        current_learning_rate = optimizer.param_groups[0]['lr']
        print(f"Epoch {str(epoch).zfill(3)}")
        print(f"Learning Rate: {current_learning_rate}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")


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

        # -- reset metrics --
        train_accuracy.reset()
        train_recall.reset()
        train_precision.reset()
        test_accuracy.reset()
        test_recall.reset()
        test_precision.reset()

        # -- alculate average loss --
        epoch_loss_train = np.mean(epoch_loss_train)
        epoch_loss_test = np.mean(epoch_loss_test)

        # --- Tensorboard ---
        # -- write to tensorboard --
        current_learning_rate = optimizer.param_groups[0]['lr']
        tensorboard.write_board(epoch, epoch_loss_train, train_acc, train_rec, train_pre, epoch_loss_test, test_acc, test_rec, test_pre, current_learning_rate)

        # -- confusion matrix --
        cm = confusion_matrix(labels.cpu().numpy(), torch.argmax(preds, dim=1).cpu().numpy())
        tensorboard.write_confusion_matrix(epoch, cm)

        # -- parameter histogram --
        tensorboard.write_parameter_histogram(model, epoch)

        # -- save model --
        modelsPath = os.path.join(output_folder, 'models')
        if config['safe_model_epoch'] != 0:   # if safe_model_epoch is not 0
            if epoch % config['safe_model_epoch'] == 0:   # save model after x epochs
                if not os.path.exists(modelsPath):
                    os.makedirs(modelsPath)
                torch.save(model.state_dict(), modelsPath + f"/model_{epoch}.pth")

        # --- Early Stopping ---
        # Update best validation loss and save model if validation loss improved
        if epoch_loss_test < best_test_loss and config['safe_best_model']:   # Save best model if validation loss improved and save_best_model is enabled
            best_test_loss = epoch_loss_test
            torch.save(model.state_dict(), os.path.join(output_folder, f'best_model.pth'))
            log['best_model'] = {
                'epoch': epoch,
                'accuracy': test_acc,
                'loss': best_test_loss
            }
            logging.info(f'Best model saved at epoch {epoch}.')
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1

        # Check for early stopping
        if early_stopping_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch} as validation loss did not improve for {patience} epochs.')
            break

        # --- Logging ---
        # -- print epoch summary --
        end_time = datetime.datetime.now()  # Record end time of epoch
        epoch_time = end_time - start_time  # Calculate epoch time

        logging.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Epoch Time: {epoch_time}")
        logging.info(f"TRAIN\t loss: {epoch_loss_train:.4f} \t acc: {train_acc * 100:.4f}")
        logging.info(f"TEST\t loss: {epoch_loss_test:.4f} \t acc: {test_acc * 100:.4f}")
        logging.info('')

        # --- Update lr sheduler ---
        lr_scheduler.step()

    tensorboard.writer.flush()
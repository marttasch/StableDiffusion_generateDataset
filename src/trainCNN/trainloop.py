import torch
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import numpy as np
import datetime
import os
import json
import sys

import gotifyHandler as gotify
from mts_utils import *
import prettytable as pt

      
# --- Function for Trainloop ---
def trainloop(model, config, device, class_names, train_set, test_set, output_folder, mean, std, datasetName, modelSelection, tensorboard  , logging):
    model = model.to(device)

    # --- init ---
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = torch.nn.CrossEntropyLoss()
    tensorboard.init_tensorboard()

    # --- Log ---
    log = {}
    log['dataset'] = datasetName
    log['config'] = config
    log['class_names'] = class_names
    log['training_set'] = [str(train_set)]
    log['test_set'] = [str(test_set)]
    log['mean'] = [str(mean)]
    log['std'] = [str(std)]
    log['device'] = str(device)
    log['model'] = model.__class__.__name__
    log['tuning_method'] = 'finetuning' if modelSelection == 1 else 'feature extraction'
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

    # epoch log
    epoch_log = {}


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
    startTime = time.time()  # Record start time of training loop
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Record start time of epoch

        print('')
        logging.info(f"Epoch {str(epoch).zfill(3)}")
        
        # Logging
        epoch_loss_train, epoch_loss_test = [], []    # store loss for each epoch
        current_learning_rate = optimizer.param_groups[0]['lr']

        # --- Training ---
        for imgs, labels in train_loader:
            model.train()   # setup model for training
            imgs, labels = imgs.to(device), labels.to(device)   # move data to device

            # -- forward pass --
            outputs = model(imgs)               # get predictions

            # Check if the model is Inception
            if model.__class__.__name__ == 'Inception3':
                main_outputs, aux_outputs = outputs   # Inception has two outputs
                loss_train = criterion(main_outputs, labels) + 0.4 * criterion(aux_outputs, labels)   # calculate loss
                metrics_outputs = main_outputs
            else:
                # For other models, there's only one output to compute the loss on
                loss_train = criterion(outputs, labels)   # calculate loss
                metrics_outputs = outputs

            # -- backward pass --
            optimizer.zero_grad()   # reset gradients
            loss_train.backward()   # calculate gradients
            optimizer.step()       # update weights

            # -- update metrics --
            t_a = train_accuracy(metrics_outputs, labels)
            train_recall(metrics_outputs, labels)
            train_precision(metrics_outputs, labels)

            epoch_loss_train.append(loss_train.item())   # store loss


        # --- Testing ---
        with torch.no_grad():   # no need to track gradients for testing
            for imgs, labels in test_loader:   # iterate over test data
                model.eval()   # set model to evaluation mode
                imgs, labels = imgs.to(device), labels.to(device)   # move data to device

                # -- forward pass --
                outputs = model(imgs)
                loss_test = criterion(outputs, labels)

                # -- update metrics --
                test_accuracy(outputs, labels)
                test_recall(outputs, labels)
                test_precision(outputs, labels)

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

        #print(f"epoch_loss_train_mean: {epoch_loss_train}")

        # --- Tensorboard ---
        # -- write to tensorboard --
        current_learning_rate = optimizer.param_groups[0]['lr']
        tensorboard.write_board(epoch, epoch_loss_train, train_acc, train_rec, train_pre, epoch_loss_test, test_acc, test_rec, test_pre, current_learning_rate)

        # -- confusion matrix --
        #cm = confusion_matrix(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
        #tensorboard.write_confusion_matrix(epoch, cm)

        # -- parameter histogram --
        #tensorboard.write_parameter_histogram(model, epoch)

        # -- save model --
        modelsPath = os.path.join(output_folder, 'models')
        if config['safe_model_epoch'] != 0:   # if safe_model_epoch is not 0
            if epoch % config['safe_model_epoch'] == 0:   # save model after x epochs
                if not os.path.exists(modelsPath):
                    os.makedirs(modelsPath)
                torch.save(model.state_dict(), modelsPath + f"/model_{epoch}.pth")

        # --- Logging ---
        # -- print epoch summary --
        epoch_time = get_TimeElapsed(epoch_start_time)

        #logging.info(f"Epoch Time: {epoch_time}")
        #logging.info(f"TRAIN\t loss: {epoch_loss_train:.4f} \t acc: {train_acc * 100:.4f}")
        #logging.info(f"TEST\t loss: {epoch_loss_test:.4f} \t acc: {test_acc * 100:.4f}")

        # pretty table
        tableEpochMetrics = pt.PrettyTable()
        tableEpochMetrics.field_names = ["", "Loss", "Accuracy", "Recall", "Precision"]
        tableEpochMetrics.add_row(["Train", f"{epoch_loss_train:.4f}", f"{train_acc:.4f}", f"{train_rec:.4f}", f"{train_pre:.4f}"])
        tableEpochMetrics.add_row(["Test", f"{epoch_loss_test:.4f}", f"{test_acc:.4f}", f"{test_rec:.4f}", f"{test_pre:.4f}"])
        logging.info(f"\n{tableEpochMetrics}")


        # epoch log
        epoch_log[epoch] = {
            'metrics': {
                'train_loss': str(epoch_loss_train.item()),
                'test_loss': str(epoch_loss_test.item()),
                'train_acc': str(train_acc.item()),
                'test_acc': str(test_acc.item()),
                'train_recall': str(train_rec.item()),
                'test_recall': str(test_rec.item()),
                'train_precision': str(train_pre.item()),
                'test_precision': str(test_pre.item()),
                'time': str(epoch_time),
            },
            'hyperparameters': {
                'learning_rate': current_learning_rate,
            }
        }
        # write epoch log to json
        with open(os.path.join(output_folder, 'epoch_log.json'), 'w') as f:
            json.dump(epoch_log, f, indent=4)

        # -- write log to json --
        log['finished_epochs'] = epoch
        with open(os.path.join(output_folder, 'log.json'), 'w') as f:
            json.dump(log, f, indent=4)

        tensorboard.plot_tensorboard_data()
        #logging.info(f"plotted tensorboard data for epoch {epoch}")

        # --- Early Stopping ---
        # Update best validation loss and save model if validation loss improved
        if epoch_loss_test < best_test_loss and config['safe_best_model']:   # Save best model if validation loss improved and save_best_model is enabled
            best_test_loss = epoch_loss_test
            torch.save(model.state_dict(), os.path.join(output_folder, f'best_model.pth'))
            log['best_model'] = {
                'epoch': epoch,
                'accuracy': str(test_acc.item()),
                'loss': str(epoch_loss_test),
            }
            logging.info(f'Best model saved at epoch {epoch}.')
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1

        # Check for early stopping
        if early_stopping_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch} as test loss did not improve for {patience} epochs.')
            break

        # --- Update lr sheduler ---
        lr_scheduler.step(epoch_loss_test if hasattr(lr_scheduler, 'step') else None)

    # --------------------------------
    # ------- END training loop ------
    # --------------------------------
    tensorboard.writer.flush()   # flush tensorboard writer

    # send gotify message
    try:
        gotify.send_message(
            title='CNN Training finished',
            message=f"""Finished {epoch} epochs in {get_TimeElapsed(startTime)}. \nBest model at epoch {log['best_model']['epoch']} with accuracy {log['best_model']['accuracy']} and loss {log['best_model']['loss']}.""",
            priority=7
        )
    except Exception as e:
        logging.error(f"Gotify message could not be sent: {e}")

    # ---- Rename Folder ----
    # rename folder with "finished" at the end
    bestEpoch = int(log['best_model']['epoch'])
    bestAcc = float(log['best_model']['accuracy'])
    bestLoss = float(log['best_model']['loss'])

    folderPraefix = f'_finished_EP-{bestEpoch}_ACC-{bestAcc:.4f}_LOSS-{bestLoss:.4f}'

    # write finished file
    with open(os.path.join(output_folder, folderPraefix), 'w') as f:
        f.write('Training finished.')
        f.write(f'Epochs: {epoch}')
        f.write(f'Best Epoch: {bestEpoch}')
        f.write(f'Best Accuracy: {bestAcc}')
        f.write(f'Best Loss: {bestLoss}')

    # rename folder
    userInput = input("\n=== Training finished. ===\nPlease close tensorboard and press Enter to rename the folder.")
    # run seperate 
    rename_folder(folder_path=output_folder, new_folder_name=output_folder + folderPraefix)

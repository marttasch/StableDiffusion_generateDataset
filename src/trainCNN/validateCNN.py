import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision
from sklearn.metrics import confusion_matrix


import prettytable as pt
import numpy as np
import json
import os
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from trainingFunctions import calc_mean_and_std, load_validation_data

# ======== CONFIGURATION ========
batch_size = 16
datasetsFolder = r"D:\code\Bachelorarbeit\xx_code_BA\datasets"   # if dataset folder not passed as argument
# ==============================

 
# --- Define validation function ---
def validate_model(model, device, class_names, val_set, batch_size: int, output_folder, logging):
    model = model.to(device)
    model.eval()   # set model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    # --- Metrics ---
    val_accuracy = MulticlassAccuracy(num_classes=len(class_names)).to(device)
    val_recall = MulticlassRecall(num_classes=len(class_names)).to(device)
    val_precision = MulticlassPrecision(num_classes=len(class_names)).to(device)

    # --- Validation Loop ---
    epoch_loss_val = []
    all_targets = []
    all_preds = []
    with torch.no_grad():   # no need to track gradients for validation
        for imgs, labels in val_loader:   # iterate over validation data
            imgs, labels = imgs.to(device), labels.to(device)   # move data to device

            # -- forward pass --
            outputs = model(imgs)
            loss_val = criterion(outputs, labels)

            # -- update metrics --
            val_accuracy(outputs, labels)
            val_recall(outputs, labels)
            val_precision(outputs, labels)

            epoch_loss_val.append(loss_val.item())   # store loss

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    # --- Epoch Summary ---
    # -- calculate metrics --
    val_acc = val_accuracy.compute()
    val_rec = val_recall.compute()
    val_pre = val_precision.compute()

    # -- reset metrics --
    val_accuracy.reset()
    val_recall.reset()
    val_precision.reset()

    # -- calculate average loss --
    epoch_loss_val = np.mean(epoch_loss_val)

    # --- Tensorboard ---
    #tensorboard.write_board(epoch=0, val_loss=epoch_loss_val, val_acc=val_acc, val_rec=val_rec, val_pre=val_pre)

    # -- confusion matrix --
    cm = confusion_matrix(all_targets, all_preds)
    # save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()

    #tensorboard.write_confusion_matrix(epoch=0, cm=cm)

    # --- Logging ---
    # pretty table
    tableValMetrics = pt.PrettyTable()
    tableValMetrics.field_names = ["", "Loss", "Accuracy", "Recall", "Precision"]
    tableValMetrics.add_row(["Validation", f"{epoch_loss_val:.4f}", f"{val_acc:.4f}", f"{val_rec:.4f}", f"{val_pre:.4f}"])
    logging.info(f"\n{tableValMetrics}")

    # --- Save Validation Log ---
    val_log = {
        'metrics': {
            'val_loss': str(epoch_loss_val),
            'val_acc': str(val_acc.item()),
            'val_recall': str(val_rec.item()),
            'val_precision': str(val_pre.item())
        }
    }
    with open(os.path.join(output_folder, 'validation_log.json'), 'w') as f:
        json.dump(val_log, f, indent=4)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, required=False, help='Name of the pretrained model (ResNet50, InceptionV3)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--val_set_path', type=str, required=False, help='Path to the validation set')
    parser.add_argument('--output_folder', type=str, required=False, help='Path to the output folder')
    args = parser.parse_args()

    # ===== Check Arguments =====
    # --- Pretrained Model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    folderPath = os.path.dirname(args.model_path)
    logPath = os.path.join(folderPath, 'log.json')
    if args.pretrained_model is not None:
        if args.pretrained_model.lower() not in ['resnet50', 'inceptionv3', 'inception3']:
            raise ValueError(f"Invalid pretrained model: {args.pretrained_model}")
        else:
            if args.pretrained_model.lower() in ['inceptionv3', 'inception3', 'inception', 'inception_v3']:
                pretrainedModel = 'inception_v3'
            else:
                pretrainedModel = 'resnet50'
    else:
        # get pretrained from training log
        if not os.path.exists(logPath):
            raise FileNotFoundError(f"Training log not found: {logPath}")
    
        with open(logPath, 'r') as f:
            log = json.load(f)
            pretrainedModel = log['model']
            if pretrainedModel.lower() not in ['resnet50', 'inceptionv3', 'inception3']:
                raise ValueError(f"Invalid pretrained model found in log file: {pretrainedModel}")
            else:
                if pretrainedModel.lower() in ['inceptionv3', 'inception3', 'inception', 'inception_v3']:
                    pretrainedModel = 'inception_v3'
                else:
                    pretrainedModel = 'resnet50'
    
    # --- Validation Set Path ---
    if args.val_set_path is None:
        # get validation set path from training log
        if not os.path.exists(logPath):
            raise FileNotFoundError(f"Training log not found: {logPath}")
    
        with open(logPath, 'r') as f:
            log = json.load(f)
            datasetName = log['dataset']
        
        valSetPath = os.path.join(datasetsFolder, datasetName, 'val')
    else:
        valSetPath = args.val_set_path

    if args.output_folder is None:
        outputFolder = os.path.join(folderPath, 'validation')
        os.makedirs(outputFolder, exist_ok=True)
    else:
        outputFolder = os.path.join(args.output_folder, 'validation')
        os.makedirs(outputFolder, exist_ok=True)

    # ==== END OF ARGUMENT CHECKS ====

    # --- Paths ---
    modelPath = args.model_path



    # --- Logging ---
    loggingPath = os.path.join(outputFolder, 'validate.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(loggingPath, mode='w'),
                            logging.StreamHandler()
                        ])
    logging.info(f"Validate {modelPath}")

    # --- device GPU, if available
    logging.info("Check for GPU...")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU")

    # --- calculate mean and std ---
    logging.info("Calculate mean and std...")
    mean, std = calc_mean_and_std(valSetPath)

    # --- Load Validation Set ---
    logging.info("Loading dataset...")
    val_set, class_names = load_validation_data(target_size=(512, 512), mean=mean, std=std, val_folder=valSetPath)

    # --- load model ---
    logging.info(f"Load pretrained model: {pretrainedModel}...")
    if pretrainedModel == 'resnet50':
        model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   # load pretrained model
    elif pretrainedModel == 'inception_v3':
        model_ft = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)   # load pretrained model
    else:
        logging.error(f"Pretrained model not found: {pretrainedModel}")
        exit()

    num_ftrs = model_ft.fc.in_features   # get number of input features for fc layer
    model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))  # replace fc layer with new fc layer with num of classes as output
    model_ft.to(device)

    model = model_ft
    model.load_state_dict(torch.load(modelPath))

    logging.info(f"Model loaded successfully")

    # --- Validate Model ---
    logging.info("Start validation...")
    try:
        validate_model(
            model=model,
            device=device,
            class_names=class_names,
            val_set=val_set,
            batch_size=batch_size,
            output_folder=outputFolder,
            logging=logging
        )
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        exit()
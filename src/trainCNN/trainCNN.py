### General ###
import os
import sys
import argparse
import datetime
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # For better debugging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable onednn for tensorflow

### Custom ###
# import from ../mts_utils.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mts_utils import *
import trainingFunctions as trainingFunctions
import trainloop as trainloop
import tensorboardHandler as tensorboard

### Torch ###
import torch
from torchvision import models
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision
from torchvision.transforms import transforms
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

### Tensorboard ###
from torch._C import TensorType
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4 as uu
import shutil


# ===== CONFIG =====
# Define the config
config = {  # Hyperparameter
    "lr": 0.0015, # chooses a random value between 0.001 and 0.01
    "batch_size": 16, # chooses one of these integers for batch size
    "epochs": 100,
    "patience": 6,   # early stopping
    "target_size": (512, 512),
    'safe_model_epoch': 0,   # save model after x epochs, 0 = off
    'safe_best_model': True,  # save best model
}
datasetName = 'urinal_v2'
pretrainedModel = 'resnet50'  # resnet50, inception_v3
modelSelection = 1   # 1 = finetuning, 2 = feature extraction

# ---- only change if necessary ----
datasetsFolder = 'datasets'
outputFolder = 'trainingOutput'
# ==================


# main function
def main():
    global datasetName, pretrainedModel, modelSelection, datasetsFolder, outputFolder, config, tensorboard, device

    # --- Paths ---
    # dataset path
    root_dir = os.path.join(os.getcwd(), datasetsFolder, datasetName)
    train_folder = os.path.join(root_dir, 'train')
    test_folder = os.path.join(root_dir, 'test')
    validation_folder = os.path.join(root_dir, 'val')

    if not os.path.exists(train_folder):
        logging.error(f"Train folder not found: {train_folder}")
        exit()
    if not os.path.exists(test_folder):
        logging.error(f"Test folder not found: {test_folder}")
        exit()

    # output path
    tensor_board_root = os.path.join(outputFolder, 'tensorboard')
    save_model_folder = os.path.join(outputFolder, 'models')


    # --- device GPU, if available
    logging.info("Check for GPU...")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU")

    # --- mean and std ---
    logging.info("Calculate mean and std...")
    mean, std = trainingFunctions.calc_mean_and_std(train_folder)

    #mean = [0.5368837, 0.5088144, 0.50752956]
    #std = [0.19154227, 0.19089119, 0.18817225]

    # --- Load data ---
    logging.info("Loading dataset...")

    train_set, test_set, class_names = trainingFunctions.load_data(config, mean, std, root_dir, train_folder, test_folder)   # load data
    trainingFunctions.print_class_distribution(train_set, test_set, class_names)
    
    # display some images
    logging.info(f"Please check the displayed images. To continue close the image window.")
    trainingFunctions.display_images(train_set, num_images=5)   # display some images

    # --- Model ---
    # pretrained model
    logging.info(f"Load pretrained model: {pretrainedModel}...")
    if pretrainedModel == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)             # load pretrained model
    elif pretrainedModel == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)             # load pretrained model
    else:
        logging.error(f"Pretrained model not found: {pretrainedModel}")
        exit()

    # === Finetuning ===
    if modelSelection == 1:     # finetunig
        num_ftrs = model.fc.in_features                      # get number of input features for fc layer
        model.fc = torch.nn.Linear(num_ftrs, len(class_names))     # replace fc layer with new fc layer with num of classes as output
        model = model.to(device)                          # set device to GPU, if available
        logging.info(f"prepared model for finetuning: {model._get_name()}")

    # === Feature Extraction ===
    elif modelSelection == 2:   # feature extraction
        for param in model.parameters():
            param.requires_grad = False                         # freeze all layers

        num_ftrs = model.fc.in_features                    # get number of input features for fc layer
        model.fc = torch.nn.Linear(num_ftrs, len(class_names))   # replace fc layer with new fc layer with num of classes as output
        model = model.to(device)                      # set device to GPU, if available
        logging.info(f"prepared model for feature extraction: {model._get_name()}")

    logging.info(f"loaded model: {model._get_name()}")


    # --- Tensorboard ---
    tensorboard = tensorboard.TensorBoard(tensor_board_root)

    # --- training loop ---
    # start training loop
    logging.info("Start training loop...\n")
    trainloop.trainloop(
        model=model,
        config=config,
        device=device,
        class_names=class_names,
        train_set=train_set,
        test_set=test_set, 
        output_folder=outputFolder,
        mean=mean,
        std=std,
        datasetName=datasetName,
        modelSelection=modelSelection,
        tensorboard=tensorboard,
        logging=logging
        )

def printFinalStats(starttime):
    print(f"Time elapsed: {get_TimeElapsed(starttime)}")
    

if __name__ == "__main__":
    # --- Argument Parser ---
    print("Start program...")
    parser = argparse.ArgumentParser(description='Train a CNN model')
    parser.add_argument('--dataset', type=str, help='Name of the dataset', required=False)
    parser.add_argument('--model', type=str, help='Name of the pretrained model', required=False)
    parser.add_argument('--modelSelection', type=int, help='1 = finetuning, 2 = feature extraction', required=False)
    args = parser.parse_args()

    if args.dataset:
        datasetName = args.dataset
        print(f"Dataset: {datasetName}")
    else:
        print(f"using default dataset: {datasetName}")

    if args.model:
        pretrainedModel = args.model
        if pretrainedModel not in ['resnet50', 'inception_v3']:
            print(f"Pretrained model not correct, please use: resnet50, inception_v3")
            # print help
            parser.print_help()
            exit()
        print(f"Model: {pretrainedModel}")
    else:
        print(f"using default model: {pretrainedModel}")

    if args.modelSelection:
        if args.modelSelection not in [1, 2]:
            print(f"Model selection not correct, please use: 1 = finetuning, 2 = feature extraction")
            # print help    
            parser.print_help()
            exit()
        modelSelection = args.modelSelection
        print(f"Model selection: {'finetuning' if modelSelection == 1 else 'feature extraction'}")
    else:
        print(f"using default model selection: {'finetuning' if modelSelection == 1 else 'feature extraction'}")

    # --- Paths ---
    training_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    trainMode = 'FT' if modelSelection == 1 else 'FE'
    outputFolder = os.path.join(os.getcwd(), outputFolder, f'{training_id}_{datasetName}_{pretrainedModel}_{trainMode}')
    if not os.path.exists(outputFolder):
        print(f"Create output folder: {outputFolder}")
        os.makedirs(outputFolder)

    # --- init Logger ---
    loggingPath = os.path.join(outputFolder, 'training.log')
    print(f"Logging path: {loggingPath}")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(loggingPath, mode='w'),
                            logging.StreamHandler()
                        ])
    
    # --- Start program ---
    try:
        starttime = time.time()
        main()   # start main function
    except Exception as e:
        logging.error(f"Error: {e}")
        printFinalStats(starttime)
        logging.error("Exit program")
        exit()
    except KeyboardInterrupt:
        logging.error("KeyboardInterrupt. Exit program")
        printFinalStats(starttime)
        exit()
else:
    loggingPath = os.path.join(outputFolder, 'training.log')
    print(f"Logging path: {loggingPath}")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler(loggingPath, mode='w'),
                            logging.StreamHandler()
                        ])
    



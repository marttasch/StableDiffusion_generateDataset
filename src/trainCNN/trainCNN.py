### General ###
import datetime
import numpy as np
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # For better debugging
import matplotlib.pyplot as plt

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

# init loggger
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        #logging.FileHandler(loggingPath, mode='w'),
                        logging.StreamHandler()
                    ])



# --- device GPU, if available
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logging.info("Using CPU")

# --- Config

# Define the config
config = {  # Hyperparameter
    "lr": 0.0015, # chooses a random value between 0.001 and 0.01
    "batch_size": 32, # chooses one of these integers for batch size
    "epochs": 18,
    "target_size": (512, 512),
}

traingingDatasetPath = os.path.join(os.getcwd(), 'trainingDatasets', 'testDataset')
training_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

root_dir = os.path.join(os.getcwd(), traingingDatasetPath)
train_folder = os.path.join(root_dir, 'train')
test_folder = os.path.join(root_dir, 'test')
validation_folder = os.path.join(root_dir, 'val')

tensor_board_root = os.path.join(os.getcwd(), 'trainingOutput', training_id, 'tensorboard')
save_folder = os.path.join(os.getcwd(), 'trainingOutput', training_id, 'models')

mean, std = trainingFunctions.calc_mean_and_std(train_folder)

#mean = [0.5368837, 0.5088144, 0.50752956]
#std = [0.19154227, 0.19089119, 0.18817225]



# Load data
train_set, test_set, class_names = trainingFunctions.load_data(config, mean, std, root_dir, train_folder, test_folder)

# display some images
trainingFunctions.display_images(train_set, num_images=5)


# --- Model ---
8# -- pretrained model --
# === Finetuning ===
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)             # load pretrained model
num_ftrs = model_ft.fc.in_features                      # get number of input features for fc layer
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))     # replace fc layer with new fc layer with num of classes as output
model_ft = model_ft.to(device)                          # set device to GPU, if available

# === Feature Extraction ===
model_conv = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)           # load pretrained model
for param in model_conv.parameters():
    param.requires_grad = False                         # freeze all layers

num_ftrs = model_conv.fc.in_features                    # get number of input features for fc layer
model_conv.fc = torch.nn.Linear(num_ftrs, len(class_names))   # replace fc layer with new fc layer with num of classes as output
model_conv = model_conv.to(device)                      # set device to GPU, if available

# --- model selection ---
# 1 = finetuning, 2 = feature extraction
modelSelection = 2

if modelSelection == 1:     # finetunig
    model = model_ft
elif modelSelection == 2:   # feature extraction
    model = model_conv

# --- Tensorboard ---
tensorboard = tensorboard.TensorBoard(tensor_board_root)

# --- training loop ---
logging.info("Start training loop...")
trainloop.trainloop(model, config, device, class_names, train_set, test_set, tensorboard)   # start training loop
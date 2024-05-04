from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
import os
import random 
import matplotlib.pyplot as plt



def calc_mean_and_std(train_folder):
    # Define the target size for resizing
    target_size = (224, 224)

    # Define the transformation without normalization
    t_train = Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    # Load the train dataset without normalization
    train_set = ImageFolder(root=train_folder, transform=t_train)

    # Get the pixel data from the dataset
    train_data = [image[0].numpy() for image in train_set]

    # Compute the mean and standard deviation using numpy
    mean = np.mean(train_data, axis=(0, 2, 3))
    std = np.std(train_data, axis=(0, 2, 3))

    print("Mean:", mean)
    print("Standard Deviation:", std)

    return mean, std


def load_data(config, mean, std, root_dir, train_folder, test_folder): # Directory as argument
  target_size = config['target_size']

  # data augmentation
  t_train = Compose([
                    #BoundingTransform(),
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=30),
                    transforms.Grayscale(3),
                    transforms.GaussianBlur(kernel_size=3),
                    #transforms.RandomCrop(size=(224, 224), padding=4),
                     ])

  t_test = Compose([transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

  os.chdir(root_dir) # defining workspace
  train_set = ImageFolder(root=train_folder, transform=t_train)
  test_set = ImageFolder(root=test_folder, transform=t_test)


  # --- get class names
  class_names = train_set.classes

  return train_set, test_set, class_names

def display_images(dataset, num_images=5):
  # Get random indices for selecting images
  random_indices = random.sample(range(len(dataset)), num_images)

  # Create a grid of subplots to display the images
  fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

  # Iterate over the selected images and display them
  for i, idx in enumerate(random_indices):
      image, _ = dataset[idx]
      image = image.permute(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

      # Normalize pixel values to [0, 1]
      image = (image - image.min()) / (image.max() - image.min())

      axes[i].imshow(image)
      axes[i].axis('off')

  plt.tight_layout()
  plt.show()
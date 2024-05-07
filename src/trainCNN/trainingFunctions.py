from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
import numpy as np
import os
import random 
import matplotlib.pyplot as plt
from prettytable import PrettyTable



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

def print_class_distribution(train_set, test_set, class_names):
  # Get the number of images per class in the training set
  train_class_counts = {class_name: sum([label == i for label in train_set.targets]) for i, class_name in enumerate(class_names)}

  # Get the number of images per class in the test set
  test_class_counts = {class_name: sum([label == i for label in test_set.targets]) for i, class_name in enumerate(class_names)}

  # Print the class distribution for the training set
  print("Class Distribution's for Training and Test Set")
  table = PrettyTable(["Class Name", "Number of Images (Train) / %", "Number of Images (Test) / %"])
  for class_name in class_names:
    num_images_train = train_class_counts[class_name]
    percentage_train = num_images_train / len(train_set) * 100
    num_images_test = test_class_counts[class_name]
    percentage_test = num_images_test / len(test_set) * 100
    table.add_row([class_name, f"{num_images_train} / {percentage_train:.2f}%", f"{num_images_test} / {percentage_test:.2f}%"])
  print(table)

def plot_metrics(metrics, title='Metrics', xlabel='Epoch', ylabel='Value', output_path='./' ,output_name='metrics_plot'):
  # Create a figure and axis for plotting
  fig, ax = plt.subplots(figsize=(12, 8))

  # Plot the metrics
  for metric_name, values in metrics.items():
    ax.plot(range(1, len(values) + 1), values, label=metric_name)

  # Add labels and title
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  # Add a legend
  ax.legend()

  save_path = os.path.join(output_path, f'{output_name}.png')
  # Save the plot
  plt.tight_layout()
  plt.savefig(save_path)
  plt.show()

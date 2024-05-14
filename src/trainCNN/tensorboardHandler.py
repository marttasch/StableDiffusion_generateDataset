import datetime
from torch._C import TensorType
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4 as uu
import shutil
import os
from tensorboard import program
import subprocess

import matplotlib.pyplot as plt
import seaborn as sns

# plot tensorboard metrics
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import io
import re


class TensorBoard:
    def __init__(self, tensor_board_root: str):
        self.tensor_board_root = tensor_board_root
        print('init writer')
        self.writer = SummaryWriter(tensor_board_root)

    def init_tensorboard(self):
        self.gen_session_id()
        self.launch_tensorboard()
        self.create_launchTensorboard_script()

    def launch_tensorboard(self):
        print('launch tensorboard')
      
        tensorboard_command = ["tensorboard", "--logdir", self.tensor_board_root]
        cmd = f"start cmd /k {' '.join(tensorboard_command)}"
        print(f"\nExecuting command: {cmd}")
        os.system(cmd)

    def create_launchTensorboard_script(self):
        # Create a script to launch tensorboard, place it in the tensorboard root
        script = f"tensorboard --logdir ."
        with open(os.path.join(self.tensor_board_root, 'launch_tensorboard.bat'), 'w') as f:
            f.write(script)

    def gen_session_id(self) -> str:
        day = str(datetime.date.today().day).rjust(2, '0')
        month = str(datetime.date.today().month).rjust(2, '0')
        year = str(datetime.date.today().year)[-2:]
        hour = str(datetime.datetime.now().hour).rjust(2, '0')
        minute = str(datetime.datetime.now().minute).rjust(2, '0')
        self.session_id = f'{day}-{month}-{year}_{hour}-{minute}_{str(uu())[-8:]}'
        print(f"Session ID: {self.session_id}")
        return self.session_id


    def write_board(self, epoch: int, train_loss: int, train_accuracy, train_recall, train_precision, test_loss, test_accuracy, test_recall, test_precision, current_learning_rate: float):
      
        # Calc metric and add scalars
        self.writer.add_scalars(f'Loss/{self.session_id}', {'Train': train_loss, 'Test': test_loss}, epoch)

        self.writer.add_scalars(f'Accuracy/{self.session_id}', {'Train': train_accuracy, 'Test': test_accuracy}, epoch)

        self.writer.add_scalars(f'Recall/{self.session_id}', {'Train': train_recall, 'Test': test_recall}, epoch)

        self.writer.add_scalars(f'Precision/{self.session_id}', {'Train': train_precision, 'Test': test_precision}, epoch)

        self.writer.add_scalar(f'Learning Rate/{self.session_id}', current_learning_rate, epoch)

        self.writer.flush()

    def write_confusion_matrix(self, epoch: int, confusion_matrix):
        # log confusion matrix

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.writer.add_figure('Confusion Matrix', plt.gcf(), global_step=epoch)
        self.writer.flush()

    def write_parameter_histogram(self, model, epoch):
        # log parameter distributions

        for name, param in model.named_parameters():
            plt.hist(param.flatten().detach().cpu().numpy(), bins=50, alpha=0.5, label=name)
            plt.legend()
            plt.title('Parameter Histograms')
            self.writer.add_figure('Parameter Histograms', plt.gcf(), global_step=epoch)
            self.writer.flush()




# ===== Plot TensorBoard Metrics =====

    def load_all_events(self, logdir=None):
        if logdir is None:
            logdir = self.tensor_board_root
        accumulators = []
        for root, dirs, files in os.walk(logdir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    path = os.path.join(root, file)
                    type = root.split('_')[-1]
                    ea = event_accumulator.EventAccumulator(path,
                        size_guidance={
                            event_accumulator.SCALARS: 0,
                            event_accumulator.IMAGES: 0,
                            event_accumulator.HISTOGRAMS: 0
                        }
                    )
                    ea.Reload()
                    accumulator = {
                        'ea': ea,
                        'path': path,
                        'type': type
                    }
                    accumulators.append(accumulator)
        return accumulators

    def plot_tensorboard_data(self, logdir=None):
        filter_images = True
        image_tag_whitelist = ['confusion matrix']

        if logdir is None:
            # self.tensor_board_root, one level up, + 'images-plots'
            logdir = os.path.join(self.tensor_board_root, '..', 'images-plots')

        # Prepare directories for plots and images
        plot_dir = os.path.join(logdir, 'plots')
        image_dir = os.path.join(logdir, 'images')
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Define the regex pattern
        #pattern = r"^(\w+)_(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2})_([a-f0-9]+)_(Train|Test)$"
        pattern = r"^(.+)/(\d{2}-\d{2}-\d{2})_(\d{2}-\d{2})_([a-f0-9]+)$"

        # Initialize the Event Accumulator with only scalars and images
        accumulators = self.load_all_events(logdir)

        # Extract scalar data
        metrics = {}
        for accumulator in accumulators:
            ea = accumulator['ea']
            data_type = accumulator['type']
            path = accumulator['path']

            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                match = re.match(pattern, tag)
                #print(f"\nTag: {tag}")
                if match:
                    metric_name, _, _, _ = match.groups()
                    #print(f"Metric Name: {metric_name}, Type: {type}")
                    if metric_name.lower() in ['accuracy', 'loss', 'recall', 'precision']:
                        events = ea.Scalars(tag)
                        values = [e.value for e in events]
                        steps = [e.step for e in events]
                        full_tag = f'{metric_name}_{data_type}'
                        #full_tag = f'{metric_name}'
                        metrics[full_tag] = pd.DataFrame({'Step': steps, 'Value': values})
                    # print(f"Values: {metrics[full_tag]}")
                    elif metric_name.lower() in ['learning rate']:
                        events = ea.Scalars(tag)
                        values = [e.value for e in events]
                        steps = [e.step for e in events]
                        full_tag = f'{metric_name}'
                        metrics[full_tag] = pd.DataFrame({'Step': steps, 'Value': values})
                    
        print(f"\nExtracted metrics: {metrics.keys()}")

        # Plot metrics
        for metric in ['Accuracy', 'Loss', 'Recall', 'Precision']:
            train_tag = f'{metric}_Train'
            test_tag = f'{metric}_Test'
            
            if train_tag in metrics and test_tag in metrics:
                plt.figure(figsize=(12, 6))
                plt.plot(metrics[train_tag]['Step'], metrics[train_tag]['Value'], label='Train')
                plt.plot(metrics[test_tag]['Step'], metrics[test_tag]['Value'], label='Test')
                plt.title(f'{metric} over Training Steps')
                plt.xlabel('Training Steps')
                plt.ylabel(metric)
                plt.grid(True)
                # datapoints with label
                for i, txt in enumerate(metrics[train_tag]['Value']):
                    plt.annotate(f"{txt:.3f}", (metrics[train_tag]['Step'][i], metrics[train_tag]['Value'][i]), color='gray')
                for i, txt in enumerate(metrics[test_tag]['Value']):
                    plt.annotate(f"{txt:.3f}", (metrics[test_tag]['Step'][i], metrics[test_tag]['Value'][i]), color='gray')

                plt.legend()
                plt.savefig(os.path.join(plot_dir, f'{metric}_plot.png'))
                plt.close()
        
        # extract images
        images = {}
        for accumulator in accumulators:
            ea = accumulator['ea']
            path = accumulator['path']
            image_tags = ea.Tags()['images']
            for tag in image_tags:
                if filter_images:
                    if tag.lower() in image_tag_whitelist:
                        images[tag] = ea.Images(tag)
                else:
                    images[tag] = ea.Images(tag)

            
        print(f"\nExtracted images: {images.keys()}")

        # Export images
        for tag, image in images.items():
            for i, img in enumerate(image):
                img_data = io.BytesIO(img.encoded_image_string)
                img = plt.imread(img_data)
                plt.imshow(img)
                plt.axis('off')
                plt.title(tag)
                plt.savefig(os.path.join(image_dir, f'{tag}_{i}.png'))
                plt.close()

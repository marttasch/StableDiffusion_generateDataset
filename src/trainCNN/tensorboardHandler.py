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
import os
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator


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
        script = f"tensorboard --logdir {self.tensor_board_root}"
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


    def extract_tensorboard_data(self, log_dir=None):
        """
        Extracts data from TensorBoard log files.

        Args:
        - log_dir (str): Path to the directory containing TensorBoard event files.

        Returns:
        - data (dict): Dictionary containing metric data {metric_name: [(step1, value1), (step2, value2), ...]}.
        """
        if log_dir is None:
            log_dir = self.tensor_board_root
        data = {}

        for root, _, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out"):
                    event_path = os.path.join(root, file)
                    for event in summary_iterator(event_path):
                        for value in event.summary.value:
                            if value.tag not in data:
                                data[value.tag] = []
                            data[value.tag].append((event.step, value.simple_value))
        return data


    def plot_tensorboard_metrics(self, log_dir=None, title='TensorBoard Metrics', xlabel='Step', ylabel='Value', save_path='tensorboard_plot'):
        """
        Extracts and plots TensorBoard metrics as images.

        Args:
        - log_dir (str): Path to the directory containing TensorBoard event files.
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - save_path (str): Path to save the plot as an image.
        """
        if log_dir is None:
            log_dir = self.tensor_board_root
        metric_data = self.extract_tensorboard_data(log_dir)

        plot_types = ['accuracy', 'loss']

        for plot_type in plot_types:
            plt.figure(figsize=(12, 8))
            lines_plotted = False
            for metric, data in metric_data.items():
                steps, values = zip(*data)
                if plot_type in metric:
                    if 'train' in metric:
                        plt.plot(steps, values, label=metric)
                        lines_plotted = True
                    elif 'test' in metric:
                        plt.plot(steps, values, label=metric, linestyle='dashed')
                        lines_plotted = True

            if lines_plotted:
                plt.legend()
            plt.title(f"{plot_type.title()} {title}")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(f"{save_path}_{plot_type}.png")
            plt.close()
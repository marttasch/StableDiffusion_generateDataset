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
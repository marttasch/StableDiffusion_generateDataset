import datetime
from torch._C import TensorType
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4 as uu
import shutil
import os
from tensorboard import program


class TensorBoard:
    def __init__(self, tensor_board_root: str):
        self.tensor_board_root = tensor_board_root
        print('init writer')
        self.writer = SummaryWriter(tensor_board_root)

    def init_tensorboard(self):
        self.gen_session_id()
        #self.start_tensorboard()

    def start_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.tensor_board_root])
        url = tb.launch()
        print(f'TensorBoard started at {url}')

    def gen_session_id(self) -> str:
        day = str(datetime.date.today().day).rjust(2, '0')
        month = str(datetime.date.today().month).rjust(2, '0')
        year = str(datetime.date.today().year)[-2:]
        hour = str(datetime.datetime.now().hour).rjust(2, '0')
        minute = str(datetime.datetime.now().minute).rjust(2, '0')
        self.session_id = f'{day}-{month}-{year}_{hour}-{minute}_{str(uu())[-8:]}'
        print(f"Session ID: {self.session_id}")
        return self.session_id


    def write_board(self, epoch: int, train_loss: int, train_accuracy, train_recall, train_precision, test_loss, test_accuracy, test_recall, test_precision, session_id):
      
        # Calc metric and add scalars
        self.writer.add_scalars(f'Loss/{self.session_id}', {'Train': train_loss, 'Test': test_loss}, epoch)

        self.writer.add_scalars(f'Accuracy/{self.session_id}', {'Train': train_accuracy, 'Test': test_accuracy}, epoch)

        self.writer.add_scalars(f'Recall/{self.session_id}', {'Train': train_recall, 'Test': test_recall}, epoch)

        self.writer.add_scalars(f'Precision/{self.session_id}', {'Train': train_precision, 'Test': test_precision}, epoch)

        self.writer.flush()
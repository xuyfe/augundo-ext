import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TemplateStereoDataset(Dataset):
    '''
    Template dataset for stereo depth estimation with the AugUndo pipeline.

    The stereo AugUndo training loop expects each batch to contain 4 images:
        - Left image at time t
        - Right image at time t
        - Left image at time t+1 (temporal neighbor)
        - Right image at time t+1 (temporal neighbor)

    If your model does not use temporal frames (temporal_loss_weight=0),
    you can return dummy tensors for the t+1 frames.

    Arg(s):
        data_path : str
            root path to your dataset
        filenames_file : str
            path to a text file listing training samples
        input_height : int
            target image height after resizing
        input_width : int
            target image width after resizing
    '''

    def __init__(self,
                 data_path,
                 filenames_file,
                 input_height=256,
                 input_width=512):

        self.data_path = data_path
        self.input_height = input_height
        self.input_width = input_width

        # TODO: Load your file list
        # Example: each line contains paths to left_t, right_t, left_t1, right_t1
        # with open(filenames_file, 'r') as f:
        #     self.filenames = f.readlines()
        self.filenames = []

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        '''
        Returns:
            left_t : torch.Tensor[float32]
                3 x H x W left image at time t, values in [0, 1]
            right_t : torch.Tensor[float32]
                3 x H x W right image at time t, values in [0, 1]
            left_t1 : torch.Tensor[float32]
                3 x H x W left image at time t+1, values in [0, 1]
            right_t1 : torch.Tensor[float32]
                3 x H x W right image at time t+1, values in [0, 1]
        '''

        # TODO: Load and preprocess your images
        # 1. Read the 4 images (left_t, right_t, left_t1, right_t1)
        # 2. Resize to (input_height, input_width)
        # 3. Convert from uint8 [0, 255] to float32 [0, 1]
        # 4. Convert from HWC to CHW format
        #
        # Example:
        #   img = cv2.imread(path)
        #   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #   img = cv2.resize(img, (self.input_width, self.input_height))
        #   img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        left_t = torch.zeros(3, self.input_height, self.input_width)
        right_t = torch.zeros(3, self.input_height, self.input_width)
        left_t1 = torch.zeros(3, self.input_height, self.input_width)
        right_t1 = torch.zeros(3, self.input_height, self.input_width)

        return left_t, right_t, left_t1, right_t1


def create_template_dataloader(data_path, filenames_file, input_height, input_width,
                               batch_size, num_threads=4, shuffle=True):
    '''
    Creates a DataLoader for the template stereo dataset.

    This function is called from train_stereo_depth_completion.py.
    You should register it in stereo_depth_completion.py alongside
    create_bdf_dataloader and create_unos_dataloader.

    Arg(s):
        data_path : str
            root path to dataset
        filenames_file : str
            path to filenames text file
        input_height : int
            input image height
        input_width : int
            input image width
        batch_size : int
            batch size
        num_threads : int
            number of data loading workers
        shuffle : bool
            whether to shuffle the dataset

    Returns:
        torch.utils.data.DataLoader : training dataloader
    '''

    dataset = TemplateStereoDataset(
        data_path=data_path,
        filenames_file=filenames_file,
        input_height=input_height,
        input_width=input_width)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        drop_last=True)

    return dataloader

# 自定义数据集
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from torchvision.io import read_image

class MyDataset(Dataset):
    def __init__(self, input_folder, truth_folder, transform=None):
        self.input_folder = input_folder
        self.truth_folder = truth_folder
        self.transform = transform
        
        self.image_filenames = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(truth_folder, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.image_filenames[idx])
        truth_path = os.path.join(self.truth_folder, self.image_filenames[idx])
        
        input_image = read_image(input_path) / 255.0
        truth_image = read_image(truth_path) / 255.0

        if self.transform:
            input_image = self.transform(input_image)
            truth_image = self.transform(truth_image)

        return input_image, truth_image, self.image_filenames[idx]

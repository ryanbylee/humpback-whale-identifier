import torch
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image

class WhaleDataset(torch.utils.data.Dataset):

    def __init__(self, mapping_file, img_directory, transform = None, target_transform = None):
        self.labels = pd.read_csv(mapping_file)
        self.img_directory = img_directory
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, index):
        path = os.path.join(self.img_directory, self.labels.iloc[index, 0])
        image = read_image(path)
        label = self.labels.iloc[index, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.labels)
    

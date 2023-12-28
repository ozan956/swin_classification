# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Implement dataset loading logic here
        # Example: Load images and labels from 'data_dir'
        self.data_dir = data_dir
        self.transform = transform

        # Create a list to store file paths and corresponding labels
        self.file_paths = []
        self.labels = []

        # Get the class names from the subdirectories in the data directory
        class_names = sorted(os.listdir(data_dir))

        # Map class names to numerical labels
        self.class_to_label = {class_name: i for i, class_name in enumerate(class_names)}

        # Iterate through the classes and gather file paths and labels
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                self.file_paths.append(file_path)
                self.labels.append(label)


    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Return a data sample and its label
        # Example: Read an image and its corresponding label
        img = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

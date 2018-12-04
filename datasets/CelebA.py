import torchvision.datasets as datasets
from torch.utils import data
from PIL import Image
import torch
import os
import random

class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    
    def __init__(self,
    image_dir,
    attr_path,
    chosen_attributes,
    transform, mode = 'train'):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.chosen_attributes = chosen_attributes
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        # Skip Header + empty line
        lines = lines[2:]

        # TODO make proper seed, I guess this constant seed is to always get the same test data
        random.seed(1234)
        random.shuffle(lines)
        
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]    # values are strings '-1', '1'

            # Extract chosen_attributes only
            # NOTE Some images will have none of the chosen_attributes labels and the hotOneVector will be all zeros
            hotOneVector = []
            for attr_name in self.chosen_attributes:
                idx = self.attr2idx[attr_name]
                hotOneVector.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, hotOneVector])
            else:
                self.train_dataset.append([filename, hotOneVector])
            
        print('Finished preprocessing the CelebA dataset...')   
        
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images
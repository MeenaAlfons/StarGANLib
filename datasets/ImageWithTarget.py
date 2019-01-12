import torchvision.datasets as datasets
from torch.utils import data
from PIL import Image
import torch
import os
import random

class ImageWithTarget(data.Dataset):
    """Dataset class for ..."""
    
    def __init__(self,
    image_dir,
    target_path,
    transform):
        """Initialize and preprocess."""
        self.image_dir = image_dir
        self.target_path = target_path
        self.transform = transform
        self.dataset = []

        self.preprocess()
        self.num_images = len(self.dataset)
        
    def preprocess(self):
        """Preprocess target file."""
        lines = [line.rstrip() for line in open(self.target_path, 'r')]
 
        # Skip Header line
        lines = lines[1:]

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            
            target = []
            for value in values:
                target.append(value == '1')

            print(filename, target)

            self.dataset.append([filename, target])

        print('Finished preprocessing images with targets...')
        
    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename, target = self.dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(target), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images
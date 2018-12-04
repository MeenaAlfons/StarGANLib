
import torch
from torch.utils.data import Dataset

class HotOneWrapper(Dataset):
    
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __validate__(self, class_index):
        if (class_index < 0) or (class_index >= self.num_classes):
            raise Exception(
                "class_index values must >=0 and < {}".format(self.num_classes))

    def __classIndexToHotOneVector__(self, class_index, num_classes):
        hotOneVector = torch.zeros(num_classes)
        hotOneVector[class_index] = 1
        return hotOneVector

    def __getitem__(self, index):
        image, class_index = self.dataset.__getitem__(index)
        self.__validate__(class_index)
        hotOneVector = self.__classIndexToHotOneVector__(class_index, self.num_classes)
        return image, hotOneVector

    def __len__(self):
        """Return the number of images."""
        return self.dataset.__len__()

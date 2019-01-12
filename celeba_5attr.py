import starganlib as sg
from datasets.CelebA import CelebA
from datasets.HotOneWrapper import HotOneWrapper
import os
import torch
from torch.utils import data

import torchvision.datasets as datasets

from torchvision import transforms as T

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)

    crop_size=178
    image_size=128
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    image_dir = "E:/AlexU/master/Computer Vision - Marwan/project/stargan/data/CelebA_nocrop/images"
    attr_path = "E:/AlexU/master/Computer Vision - Marwan/project/stargan/data/list_attr_celeba.txt"
    chosen_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    celeba = CelebA(image_dir, attr_path, chosen_attributes, transform=transform)

    hyper_parameters = sg.HyperParamters()
    stargan = sg.StarGAN(hyper_parameters)
    stargan.addDataset(celeba, 5)
    training_parameters = sg.TrainingParams(
        resume_iter=200000,
        num_iters=200001
    )
    stargan.train(training_parameters)

    print("DONE ........")
    print(torch.__path__)
import producer as P
from datasets.CelebA import CelebA
from datasets.ImageWithTarget import ImageWithTarget
import os
import torch
from torch.utils import data
from PIL import Image

import torchvision.datasets as datasets

from torchvision import transforms as T

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    image_dir = os.path.join(dirname, './data/image_with_target')
    target_path = os.path.join(image_dir, 'targets.txt')
    model_save_dir = os.path.join(dirname, './pretrained_model')
    results_dir = os.path.join(dirname, './results')
    save_intermediate_dir = os.path.join(dirname, './intermediate')

    crop_size=178
    image_size=128
    transform = []
    # transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    imageWithTargetDataset = ImageWithTarget(
        image_dir=image_dir,
        target_path=target_path,
        transform=transform
    )

    # Only one dataset with 5 labels
    dataset_labels_sizes = [5]

    producer = P.Poducer(
        dataset_labels_sizes=dataset_labels_sizes,
        model_save_iter=200000,
        model_save_dir=model_save_dir,
        results_dir=results_dir,
        batch_size=6,
        save_intermediate=True,
        save_intermediate_dir=save_intermediate_dir
        )

    producer.produce(imageWithTargetDataset)

    print("DONE ........")

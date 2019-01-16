import os
import time
import datetime

import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Generator

class Poducer(object):
    """    """
    
    def __init__(self,
        dataset_labels_sizes,
        image_size=128,
        batch_size=3,
        num_workers=1,
        log_step=1,
        results_dir='./samples',
        model_save_dir='./model',
        model_save_iter=0,
        save_intermediate=False,
        save_intermediate_dir='./intermediate',
        save_intermediate_prefix='layer'
        ):
        """Contructor"""
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.log_step = log_step
        self.results_dir = results_dir
        self.model_save_dir = model_save_dir
        self.model_save_iter = model_save_iter

        self.save_intermediate = save_intermediate
        self.save_intermediate_dir = save_intermediate_dir
        self.save_intermediate_prefix = save_intermediate_prefix

        self.dataset_labels_sizes = dataset_labels_sizes
        self.num_datasets = len(dataset_labels_sizes)
        self.total_num_labels = sum(dataset_labels_sizes)
        self.model_ready = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()
        self.restore_model(model_save_dir, model_save_iter)

        gDict = self.G.state_dict()
        for key in gDict:
            print("{}.shape = {}".format(key, gDict[key].shape))

    def build_model(self):
        """ """
        # Build generator
        label_dimension = self.total_num_labels
        if self.num_datasets > 1:
            label_dimension += self.num_datasets

        self.G = Generator(
            self.image_size, 
            label_dimension,
            save_intermediate=self.save_intermediate,
            save_intermediate_dir=self.save_intermediate_dir,
            save_intermediate_prefix=self.save_intermediate_prefix
            )

        self.G.to(self.device)
        self.model_ready = True

    def restore_model(self, model_save_dir, model_save_iter):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(model_save_iter))
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(model_save_iter))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def produce(self, imageWithTargetDataset):
        """ """
        if not self.model_ready :
            print("Model is not ready")
            return

        data_loader = data.DataLoader(
            dataset=imageWithTargetDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        data_iterator = iter(data_loader)
        while True:
            try:
                start_time = time.time()

                images, targets, filenames = next(data_iterator)
                images = images.to(self.device)
                output_images = self.G(images, targets)

                elapsed_time = time.time() - start_time
                elapsed_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]
                print("Elapsed [{}]".format(elapsed_time))

                self.save(output_images, filenames)
            except StopIteration:
                break

    def save(self, images, filenames):
        for i in range(len(filenames)):
            result_path = os.path.join(self.results_dir, filenames[i])
            save_image(self.denorm(images[i].data.cpu()), result_path, nrow=1, padding=0)


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

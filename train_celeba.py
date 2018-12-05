import os
import argparse
from torchvision import transforms as T

import starganlib as sg
from datasets.CelebA import CelebA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')

    config = parser.parse_args()

    crop_size=178
    image_size=128
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dirname = os.path.dirname(__file__)
    image_dir = os.path.join(dirname, "./data/CelebA_nocrop/images")
    attr_path = os.path.join(dirname, "./data/list_attr_celeba.txt")
    chosen_attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    celeba = CelebA(image_dir, attr_path, chosen_attributes, transform=transform)

    hyper_parameters = sg.HyperParamters(
        image_size=image_size,
        batch_size=config.batch_size,
        n_critic=5,
        num_workers=1,
        mode='train',
        g_lr=0.0001,
        g_conv_dim=64,
        g_repeat_num=6,
        d_lr=0.0001,
        d_conv_dim=64,
        d_repeat_num=6,
        adam_betas=(0.5,0.999),
        lambda_cls=1,
        lambda_rec=10,
        lambda_gp=10

    )
    stargan = sg.StarGAN(hyper_parameters)
    stargan.addDataset(celeba, 5)

    train_params = sg.TrainingParams(
        resume_iter=0,
        num_iters=20000,
        num_iters_decay=100000,
        lr_update_step=1000,
        log_step=10,
        sample_step=1000,
        model_save_step=10000,
        sample_dir='./samples',
        model_save_dir='./model'
    )
    stargan.train(train_params)

    print("DONE ........")
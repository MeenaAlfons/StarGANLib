import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision.utils import save_image
import math

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self,
    image_size=128,
    c_dim=5,
    save_intermediate=False,
    save_intermediate_dir='./intermediate',
    save_intermediate_prefix='layer'
    ):
        super(Generator, self).__init__()
        self.save_intermediate_dir = save_intermediate_dir
        self.save_intermediate_prefix = save_intermediate_prefix
        
        repeat_num = int(np.log2(image_size)) - 1
        first_conv_dim = 2**repeat_num

        layers = []
        layers.append(nn.Conv2d(3+c_dim, first_conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(first_conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = first_conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        if save_intermediate:
            for index, layer in enumerate(layers):
                layer.register_forward_hook(self.printLayerHook(index))

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

    def printLayerHook(self, layerIndex):
        def hook(module, input_, output):
            if layerIndex == 0:
                self.saveLayer(layerIndex, input_[0])
            self.saveLayer(layerIndex+1, output)
        return hook 
    
    def saveLayer(self, layerIndex, data):
        # layer associated with the first image of the batch
        layersOfFirstItem = data[0]
        if layersOfFirstItem.shape[0] == 3 :
            # also save it as RGB image
            asRGB = layersOfFirstItem.unsqueeze(0)
            image_name = "{}-{}-rgb.jpg".format(self.save_intermediate_prefix, layerIndex)
            self.saveImage(asRGB, image_name)
        
        layers_1_channel = layersOfFirstItem.unsqueeze(1)
        layers_3_channel = layers_1_channel.repeat(1, 3, 1, 1)
        
        image_name = "{}-{}.jpg".format(self.save_intermediate_prefix, layerIndex)
        self.saveImage(layers_3_channel, image_name)
        
    def saveImage(self, tensor, image_name):
        image_path = os.path.join(self.save_intermediate_dir, image_name)
        w = int(math.sqrt(tensor.shape[0] * 16 / 9))
        save_image(
            self.denorm(tensor), image_path,
            nrow=w,
            padding=0)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, c_dim=5):
        super(Discriminator, self).__init__()
        repeat_num = int(np.log2(image_size)) - 1
        first_conv_dim = 2**repeat_num

        layers = []
        layers += self.innerBlock(3, first_conv_dim)

        curr_dim = first_conv_dim
        for i in range(1, repeat_num):
            layers += self.innerBlock(curr_dim, curr_dim*2)
            curr_dim = curr_dim * 2

        # Input:  [batch_size,                  3, image_size, image_size]
        # Output: [batch_size, 2^(2*repeat_num-1),          2,          2]
        # Output: [batch_size,  image_size**2 / 8,          2,          2]
        self.main = nn.Sequential(*layers)

        # Input:  [batch_size, curr_dim, H, W]
        # Output: [batch_size,        1, H, W]
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Input:  [batch_size, curr_dim, 2, 2]
        # Output: [batch_size,    c_dim, 1, 1]
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    # Input:  [batch_size, curr_dim,   H,   W]
    # Output: [batch_size, next_dim, H/2, W/2]
    def innerBlock(self, curr_dim, next_dim):
        layers = []
        layers.append(nn.Conv2d(curr_dim, next_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        return layers

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

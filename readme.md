
# Installation

## Python 3.6

Must use python 3.6 because tensorflow-gpu is not available for python 3.7 yet.

## Python virtualenv

requirements.txt containt the necessary requirements and the CPU versions of TensorFlow and pyTorch. [See below](#Using-GPU) for more details about instaling both with GPU support.

```
pip install virtualenv
virtualenv -p python3 .env
cd .env/Scripts
activate
cd ..
cd ..
pip install -r requirements.txt
```

## Using GPU

Requires an NVIDIA GPU with compute capability >= 3.0. [Link](https://pytorch.org/docs/master/torch.html).

### Install TensorFlow

Follow instructions on [GPU Support](https://www.tensorflow.org/install/gpu) page and [installation](https://www.tensorflow.org/install/pip) page which gets down to this:

- install NVidia Driver
- install CUDA 9.0 + PATH configurations
- install cuDNN
- pip install tensorflow-gpu

### Install pyTorch

```
pip install http://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install torchvision
```

## Using CPU

### Install TensorFlow

```
pip install tensorflow
```

### Install pyTorch

```
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install torchvision
```

# Running [StarGAN](https://github.com/yunjey/StarGAN)

## CPU measures

The following command will need 11GB of RAM and 140 days to complete! Notice that `batch_size` and `log_step` affects the needed memory.

```
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --batch_size 10 --log_step 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

## GPU measures

The following command will take 44 hours to complete on Colab (NVidia Tesla K80, compute capability: 3.7).

```
python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

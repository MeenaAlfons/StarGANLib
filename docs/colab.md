# Colab Tips

Using colab for deep learning could become tricky in some ways. Here is the key tips you will need while using Colab for deep learning

## Actually a Linux Machine

The colab runtime is actually a running linux machine. You can use handy bash commands to accomplish your tasks. All oyu have to do is to use `!` before the command like the following example:

```
# Put this a code cell and run it
# This will display the contents of the current directory
!ls
```

## Install your dependencies

Depending of the libraries and frameworks you will be using, you would probably need to install them because colab does not come with them already installed. Examples:

```
# Put this in a code cell and run it
# This will install PyTorch and its vision extention
!pip3 install torch torchvision
```

## Install your code

You could use `git` to clone your code from your repository on the runtime machine of colab. Example:

```
!git clone https://github.com/MeenaAlfons/StarGANLib.git
```

The you can change the directory to the root of your directory as follows:

```
import os
os.chdir("StarGANLib") # Change the current directory

# List the files to see what is in the current directory
!ls
```

You can also pull any updates from the repo as follows:

```
# Pull new updates from the repo
!git pull

# See the new files of the current directory
!ls
```

## Run any bash script

You can run any bash script you have on your repo. Example:

```
# This will run the download script which is responsible for dowloading datasets
!bash ./scripts/download.sh celeba
```

## Mount Google Drive

This is very important tip. Deep learning applications probably take many hours or days for training. Colab only gives around 6 hours of active training of GPUs. So you would probably need to save your model regularly during training. So that when the runtime is shutdown, you can bring it back on and start from where you left.

In order to do this you need a presistent storage to save the learned model parameters. You can use your Google Drive for that by mounting it into the file system of the runtime machine and dave your model into it.

```
# This will mount your Google Drive into /gdrive directory
from google.colab import drive
drive.mount('/gdrive')

# You can use the following directories in to save the model parameters and samples which you will find in your Google Drive later
model_dir = "/gdrive/My Drive/Computer Vision/Project/starganlib/celeba/model"
samples_dir = "/gdrive/My Drive/Computer Vision/Project/starganlib/celeba/samples"
```

Running this code will give you a link. Clicking the link will ask you to give permission for the runtime machine of Colab to access your Google Drive. Once you give it the permission you will get a key like this `klfdsjfoierhrefoperkfpkls`.

You wil lalso find an input box under the link inside the Colab sheet. Copy and paste that code intor the input box and press `Enter`. Now Google Drive is mounted to the runtime machine your are working on.

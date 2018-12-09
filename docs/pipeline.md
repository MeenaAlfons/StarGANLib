
# Phase 1

Run the code provided with the paper.

**Results:**

### Using GPU

- I configured my laptop to run pyTorch and TensorFlow on my GPU and documented all the steps.
- My GPU is NVidia GeForce GT 755M with compute compatibility 3.0
- Finally pyTorch told me that this specific GPU is old and they droped it from their binaries but I can still use it if I build pyTorch from source

### Using CPU

- I configured my laptop to run pyTorch and TensorFlow on CPU and documented all the steps
- I have intel core i7-4700MQ @ 2.4 GHz with 16GB RAM
- One batch on this pc takes around 1 minute and requires a peak of 11GB of RAM (just for this python process not including other applications)
- The required 200,000 iterations to get the same results of the paper would have taken around 140 days

### Using Colab

- I configured Colab to StarGAN code with one dataset (CelebA).
- One batch would take 8 seconds with total of 44 hours to complete the 200,000 iterations
- I make it run for 20,000 epochs with a batch of 16 images for each epoch taking around 4 hours and saved the model and the sample output images.
- The sample images showed similar characteristics approaching the ones provided in the paper but with lower quality. (Because GAN hasn't been trained enough)

**Sample:**


| Original | Black Hair | Blond Hair | Brown Hair | Male | Young |
|----------|------------|------------|------------|------|-------|

<img src="images/stargan-20000-images.jpg" alt="drawing" width="500"/>

# Phase 2

Understand how GANs work and reimplement the same idea elaborated in the paper in a generic way making a library that can be used with multiple datasets. In contrast with the code provided with the paper, which has many lines related to those specific datasets used to show off the work of the paper.

**Deliverable:** Python library with documentation for how to use it to train a StarGAN

**Estimated:** two weeks

## Phase 2 Results

- The training code for StarGAN has been adapted to work on multiple datasets with different labels
- HotOneWrapper has been introduced to adapt all Dataset classes in pyTorch to work with StarGAN
- A document has been made describing different type of datasets and how to deal with them. [Working with Datasets](datasets.md).
- The new code has been run on Colab for 76,000 epochs with a batch 10 images for each epoch and the results seems promising and like the ones shown in the original paper. See below for the results.
- Essential tips for using Colab for deep learning has been documented. [Colab Tips](colab.md)

**Phase 2 Sample:**


| Original | Black Hair | Blond Hair | Brown Hair | Male | Young |
|----------|------------|------------|------------|------|-------|

<img src="images/starganlib-76000-images.jpg" alt="drawing" width="500"/>

# Phase 3

Use the developed library to get the same results of the paper.

Three variations of training are doumented in the paper:
- CelebA only
- RaFD only
- CelebA with RaFD

**Deliverable:** Sample output images and the trained model for each case.

**Estimated:** two weeks

## Phase 3 results

The results of phase 2 has already included training for 76000 epochs to get similar results. I don't think it is a good idea to repeat what they have done. I think we can skip this phase and start working on new datasets and other applications as shown in phase 4.

I would also like to study the architecture of StarGAN and document to get more insights and to be able to alter it as neccesary to give better results on other applications. So I suggest to make PHase 3 about:

- Understanding and documenting the architecture, hyper parameters and training approach of StarGAN

# Bonus Phase 4

Search for other datasets that would be useful to train a StarGAN to apply image-to-image transformations on them.

If a suitable dataset is found, try to train a StarGAN on that dataset.

## Phase 4 results

One application I had in mind is to train the network about the timing of the images. Training of images labeled with the timestamp of the image inditating at which time of the day the images is taken. The anticipated Generator should be given any image and the target timing and the output should be a new image for the same place at the target timing.

I started searching for datasets to be used in this training.

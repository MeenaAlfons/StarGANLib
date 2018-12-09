# Working with datasets

There are multiple kinds of datasets. Following are the two interesting types of datasets that stargan is supposed to work with.

## Single Attribute Dataset

This type of data set has one attribute label for each image. Examples are:

- image1 => "sad"
- image2 => "happy"

An iterator on this dataset will return the image and the attribute/label index between [0,C) where C is the number of attributes. Example:

- Assume attributes are [Angry, Sad, Happy]
- image1: (Tensor, 1)
- image1: (Tensor, 2)

## Multiple Attributes Dataset

This type of dataset has multiple attribute labels associated with each image. Examples are:

- image3 => Male, Black_Hair
- image4 => Female, Old

An iterator on this dataset will return the image and an array of size C with 0/1 values indicating which attributes are associated with this image where C is the number of attribute labels. Example:

- Assume attributes are [Male, Female, Black_Hair, Old]
- image3: (Tensor, [1,0,1,0])
- image4: (Tensor, [0,1,0,1])

## Unified Dataset Interface

StarGAN supports working with multiple attributes for each image. But in order to use those different kinds of datasets a unified interface need to exist. Here comes the need for HotOneWrapper.

## HotOneWrapper

HotOneWrapper is a class which wraps a Single Attribute Dataset. Instead of returning the label of the image as the index of the attribute, HotOneWrapper transforms this index to hot one vector with size C and returns that instead where C is the number of attribute labels. Here is above the example using HotOneWrapper:

- Assume attributes are [Angry, Sad, Happy]
- image1: (Tensor, [0, 1, 0])
- image2: (Tensor, [0, 0, 1])

## Images with no labels are allowed!

Using HotOneWrapper allows having some samples with the hot one vector with all zeros [0,0, ..., 0]. This means that this image does not have any of these labels.

Example:

- The original [StarGAN]() implementation uses dataset called "CelebA" and choose specific labels to be trained. Some of the images of this dataset does not have any of these label for which the hot one vector will be all zeros [0, 0, ..., 0]. However, they are using those images in training.


# Future Development

What if we have two datasets with the same labels?
What if some of the labels exists in both datasets?

Example: If we'd like to apply StarGAN on Day and Night images and we have multiple sets with day/night labeling. How can we use all those datasets to train the network on the Day/Night concepts?


import torch
import pandas
import matplotlib.pyplot as plt

from mnist_train.mnist_classifier import Classifier
from mnist_train.mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')
mnist_dataset_test = MnistDataset('mnist_data/mnist_test.csv')

index = 19

label, image_data_tensor, target_tensor = mnist_dataset_train.__getitem__(index)

print(label)
print(image_data_tensor)
print(target_tensor)

mnist_dataset_train.plot_image(index)
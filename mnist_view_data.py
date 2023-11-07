import sys
import matplotlib.pyplot as plt

from mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')
mnist_dataset_test = MnistDataset('mnist_data/mnist_test.csv')

RECORD = int(sys.argv[1])

label, image_data_tensor, target_tensor = mnist_dataset_train.__getitem__(RECORD)

mnist_dataset_train.plot_image(RECORD)
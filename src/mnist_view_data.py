import os
import sys

from mnist_dataset import MnistDataset

path_to_data = os.environ["DATA_CSV_MNIST"]

mnist_dataset_train = MnistDataset(path_to_data + '/mnist_train.csv')

RECORD = int(sys.argv[1])

label, image_data_tensor, target_tensor = mnist_dataset_train.__getitem__(RECORD)

mnist_dataset_train.plot_image(RECORD)
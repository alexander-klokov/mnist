import os
import sys
import pandas

import matplotlib.pyplot as plt

from mnist_classifier import Classifier
from mnist_dataset import MnistDataset

path_to_data = os.environ["DATA_CSV_MNIST"]
mnist_dataset_test = MnistDataset(path_to_data + '/mnist_test.csv')

PATH_TO_MODEL = './mnist_model'
RECORD = int(sys.argv[1])

mnist_dataset_test.plot_image(RECORD)

# infere
c = Classifier()
c.load_model()

image_data = mnist_dataset_test[RECORD][1]
output_tensor = c.forward(image_data)

pandas.DataFrame(output_tensor.detach().numpy()).plot(
    kind='bar', legend=False, ylim=(0, 1), figsize=(15, 10))

plt.show()

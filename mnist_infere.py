import sys
import pandas

import matplotlib.pyplot as plt

from mnist_train.mnist_classifier import Classifier
from mnist_train.mnist_dataset import MnistDataset

mnist_dataset_test = MnistDataset('mnist_data/mnist_test.csv')

PATH_TO_MODEL = './mnist_model'
RECORD = int(sys.argv[1])

mnist_dataset_test.plot_image(RECORD)

# infere

c = Classifier()
c.load_model()

image_data = mnist_dataset_test[RECORD][1]
output_tensor = c.forward(image_data.cuda())

pandas.DataFrame(output_tensor.cpu().detach().numpy()).plot(kind='bar', legend=False, ylim=(0,1), figsize=(15,10))

plt.show()

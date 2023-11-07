import torch

from mnist_classifier import Classifier
from mnist_dataset import MnistDataset

mnist_dataset_train = MnistDataset('mnist_data/mnist_train.csv')

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print('using cuda:', torch.cuda.get_device_name(0))
    pass

# training
epochs = 3

c = Classifier()

for i in range(epochs):
    print('training epoch', i, 'of', epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset_train:
        c.train(image_data_tensor.cuda(), target_tensor.cuda())
        pass
    pass

c.plot_progress()
c.save_model()

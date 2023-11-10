import os
import torch

from mnist_classifier import Classifier
from mnist_dataset import MnistDataset

path_to_data = os.environ["DATA_CSV_MNIST"]

mnist_dataset_train = MnistDataset(path_to_data + '/mnist_train.csv')

if torch.cuda.is_available():
    print('using cuda:', torch.cuda.get_device_name(0))
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# training
epochs = 3

C = Classifier()
C.to(device)

for i in range(epochs):
    print('training epoch', i, 'of', epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset_train:
        C.train(image_data_tensor.to(device), target_tensor.to(device))
        pass
    pass

C.plot_progress()
C.save_model()

# get some CUDA stats
print(torch.cuda.memory_summary(device, abbreviated=True))

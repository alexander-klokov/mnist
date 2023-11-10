import os
import torch

from mnist_classifier import Classifier
from mnist_dataset import MnistDataset

path_to_data = os.environ["DATA_CSV_MNIST"]
mnist_dataset_test = MnistDataset(path_to_data + '/mnist_test.csv')

score = 0
items = 0

C = Classifier()
C.load_model()

for label, image_data_tensor, target_tensor in mnist_dataset_test:
    answer = C.forward(image_data_tensor).detach().numpy()
    if (answer.argmax() == label):
        score += 1
        pass
    if items % 500 == 0:
        print('testing item', items)
    items += 1
    
    pass

print('\nr =', round(score / items, 2))
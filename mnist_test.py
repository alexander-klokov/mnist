import matplotlib.pyplot as plt

from mnist_classifier import Classifier
from mnist_dataset import MnistDataset

mnist_dataset_test = MnistDataset('mnist_data/mnist_test.csv')

score = 0
items = 0

c = Classifier()
c.load_model()

for label, image_data_tensor, target_tensor in mnist_dataset_test:
    answer = c.forward(image_data_tensor.cuda()).cpu().detach().numpy()
    if (answer.argmax() == label):
        score += 1
        pass
    if items % 500 == 0:
        print('testing item', items)
    items += 1
    
    pass

print('\nr =', round(score / items, 2))
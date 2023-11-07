import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas
import matplotlib.pyplot as plt

from mnist_train.mnist_dataset import MnistDataset

PATH_TO_MODEL = "mnist_model/mnist_model.pth"

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print('using cuda:', torch.cuda.get_device_name(0))
    pass

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
    
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

        self.counter = 0
        self.progress = []

        pass

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        pass

    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0,0.25,0.5))
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), PATH_TO_MODEL)

    def load_model(self):
        self.model.load_state_dict(torch.load(PATH_TO_MODEL))

    pass


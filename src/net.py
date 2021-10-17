"""

class weights for 2019-serialized_FEN-C.npz  are:
    [18.61048726, 3.5799615, 0.86170505, 0.45427876,
      0.37190948, 0.79824619, 3.24212056, 18.13564839]

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 17-10-2021
"""
import os
import time
import chess
import torch
import numpy as np
np.random.seed(98)

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from utils import WPRINT, EPRINT, TPRINT


def calc_class_weights(labels):
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(cw)

def fit(model, train, valid, optimizer, condition, device, epochs=10):
    WPRINT("initializing model fit with {}".format(device), str(model), True)
    for epoch in range(epochs):
        tloss, vloss = 0, 0
        for batch, (sample, label) in enumerate(train):
            sample, label = sample.to(device).float(), label.to(device).long()
            output = model(sample)
            loss = condition(output, label)
            tloss += loss.item()
            loss.backward()
            optimizer.step()
        for batch, (sample, label) in enumerate(valid):
            sample, label = sample.to(device).float(), label.to(device).long()
            output = model(sample)
            loss = condition(output, label)
            vloss += loss.item()
        TPRINT(epoch, tloss/len(train), vloss/len(valid))

"""
                #  in:  N x 13 x 8 x 8    out:  N x 20 x 8 x 8
                torch.nn.Conv2d(in_channels=13, out_channels=20, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(20),
                torch.nn.ReLU(),
                #  in:  N x 20 x 8 x 8    out:  N x 20 x 8 x 8
                torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(20),
                torch.nn.ReLU(),
                #  in:  N x 20 x 8 x 8    out:  N x 30 x 6 x 6
                torch.nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(30),
                torch.nn.ReLU(),
                #  in:  N x 30 x 6 x 6    out:  N x 30 x 6 x 6
                torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(30),
                torch.nn.ReLU(),
                #  in:  N x 30 x 6 x 6    out:  N x 40 x 4 x 4
                torch.nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                #  in:  N x 40 x 4 x 4    out:  N x 40 x 4 x 4
                torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                #  in:  N x 40 x 4 x 4    out:  N x 50 x 2 x 2
                torch.nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(50),
                torch.nn.ReLU()
                """

class ClassifierNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(ClassifierNet, self).__init__()
        self.n_classes = n_classes
        self._convolutional_layers = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=13, out_channels=40, kernel_size=(5, 5)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                )
        self._affine_layers = torch.nn.Sequential(
                torch.nn.Linear(40*2*2, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, self.n_classes)
                )

    def forward(self, x):
        x = self._convolutional_layers(x)
        x = x.flatten(start_dim=1)
        x = self._affine_layers(x) 
        return x
    
    def __str__(self):
        return 'ClassifierNet'


class RegressionNet(torch.nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__() 
        self._convolutional_layers = torch.nn.Sequential(
                #  in:  N x 13 x 8 x 8    out:  N x 20 x 8 x 8
                torch.nn.Conv2d(in_channels=13, out_channels=20, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(20),
                torch.nn.ReLU(),
                #  in:  N x 20 x 8 x 8    out:  N x 20 x 8 x 8
                torch.nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(20),
                torch.nn.ReLU(),
                #  in:  N x 20 x 8 x 8    out:  N x 30 x 6 x 6
                torch.nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(30),
                torch.nn.ReLU(),
                #  in:  N x 30 x 6 x 6    out:  N x 30 x 6 x 6
                torch.nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(30),
                torch.nn.ReLU(),
                #  in:  N x 30 x 6 x 6    out:  N x 40 x 4 x 4
                torch.nn.Conv2d(in_channels=30, out_channels=40, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                #  in:  N x 40 x 4 x 4    out:  N x 40 x 4 x 4
                torch.nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(3, 3), padding=(1, 1)),
                torch.nn.BatchNorm2d(40),
                torch.nn.ReLU(),
                #  in:  N x 40 x 4 x 4    out:  N x 50 x 2 x 2
                torch.nn.Conv2d(in_channels=40, out_channels=50, kernel_size=(3, 3)),
                torch.nn.BatchNorm2d(50),
                torch.nn.ReLU()
                )
        self._affine_layers = torch.nn.Sequential(
                torch.nn.Linear(50*2*2, 50*2),
                torch.nn.ReLU(),
                torch.nn.Linear(50*2, 1)
                )

    def forward(self, x):
        x = self._convolutional_layers(x)
        x = x.flatten(start_dim=1)
        x = self._affine_layers(x)
        return x

    def __str__(self):
        return 'RegressionNet'


class DatasetWrapper(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class DatasetChess:
    def __init__(self, shuffle=True, batch_size=4096, fname='datagen/2019-serialized_FEN-C.npz'):
        self.shuffle, self.batch_size = shuffle, batch_size
        self.X, self.Y = self._load(fname)
        self.datasets = self._train_valid_test_split()
        self.dataloaders = self._create_dataloaders()    

    def __str__(self):
        return "DatasetChess"
    
    def _load(self, fname):
        WPRINT("loading data from {}".format(fname), str(self), True)
        data = np.load(fname, allow_pickle=True)
        X, Y = data['arr_0'], data['arr_1']
        Y[Y > 3] = 1
        Y[Y < 4] = 0
        WPRINT("loaded {} samples from {}".format(X.shape, fname), str(self), True)
        return X, Y

    def _train_valid_test_split(self, p=0.6):
        WPRINT("splitting data into train/valid/test", str(self), True)
        indices, n_samples = np.arange(self.X.shape[0]), int(self.X.shape[0]*p)
        np.random.shuffle(indices)
        X, Y = self.X[indices], self.Y[indices]
        X_train, X_valid = X[:n_samples], X[n_samples:]
        Y_train, Y_valid = Y[:n_samples], Y[n_samples:]
        indices, n_samples = np.arange(X_valid.shape[0]), int(X_valid.shape[0]*0.5)
        np.random.shuffle(indices)
        X_valid, X_test = X_valid[:n_samples], X_valid[n_samples:]
        Y_valid, Y_test = Y_valid[:n_samples], Y_valid[n_samples:]
        WPRINT("done train/valid/test split\n\ttrain: {}   valid: {}   test: {}".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]), str(self), True)
        datasets = {'train': DatasetWrapper(X_train, Y_train),
                    'valid': DatasetWrapper(X_valid, Y_valid),
                    'test': DatasetWrapper(X_test, Y_test)}
        return datasets

    def _create_dataloaders(self):
        dataloaders = {k: DataLoader(self.datasets[k], batch_size=self.batch_size, shuffle=self.shuffle) for k in self.datasets.keys()}
        return dataloaders


if __name__ == "__main__":
    dataset = DatasetChess()
    model, device = ClassifierNet(2), 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters())
    # c_weights = torch.Tensor([18.61048726, 3.5799615, 0.86170505, 0.45427876, 
    #                         0.37190948, 0.79824619, 3.24212056, 18.13564839]).to(device).float()
    condition = torch.nn.CrossEntropyLoss()
    # condition = torch.nn.HuberLoss()
    # condition = torch.nn.MSELoss()
    model.to(device)
    summary(model,(13, 8, 8))
    fit(model, dataset.dataloaders['train'], dataset.dataloaders['valid'], optimizer, condition, device)


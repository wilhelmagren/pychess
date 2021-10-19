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
import torch.nn as nn
import numpy as np
np.random.seed(98)

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from utils import WPRINT, EPRINT, TPRINT


def calc_class_weights(labels):
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(cw)

def fit(model, train, valid, optimizer, condition, device, epochs=30):
    WPRINT("initializing model fit with {}".format(device), str(model), True)
    model.train()
    for epoch in range(epochs):
        tloss, tacc, vloss, vacc = 0, 0, 0, 0
        for batch, (sample, label) in enumerate(train):
            sample, label = sample.to(device).float(), label.to(device).long()
            optimizer.zero_grad()
            output = model(sample)
            loss = condition(output, label)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            _, pred = torch.max(output.data, 1)
            tacc += (pred == label).sum().item()/output.shape[0]
        for batch, (sample, label) in enumerate(valid):
            sample, label = sample.to(device).float(), label.to(device).long()
            output = model(sample)
            loss = condition(output, label)
            vloss += loss.item()
            _, pred = torch.max(output.data, 1)
            vacc += (pred == label).sum().item()/output.shape[0]
        TPRINT(epoch, tloss/len(train), tacc/len(train), vloss/len(valid), vacc/len(valid))


class ChessClassifierCNN(nn.Module):
    def __init__(self, n_classes):
        super(ChessClassifierCNN, self).__init__()
        self._conv1 = nn.Conv2d(17, 16, 3)
        self._conv2 = nn.Conv2d(16, 32, 3)
        self._conv3 = nn.Conv2d(32, 64, 3)
        self._bnorm1 = nn.BatchNorm2d(16)
        self._bnorm2 = nn.BatchNorm2d(32)
        self._bnorm3 = nn.BatchNorm2d(64)
        self._aff1 = nn.Linear(64*2*2, 256)
        self._aff2 = nn.Linear(256, 128)
        self._aff3 = nn.Linear(128, n_classes)
        self._bnorm4 = nn.BatchNorm1d(256)
        self._bnorm5 = nn.BatchNorm1d(128)
        self._droput = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = torch.relu(self._bnorm1(self._conv1(x)))
        x = torch.relu(self._bnorm2(self._conv2(x)))
        x = torch.relu(self._bnorm3(self._conv3(x)))
        x = x.flatten(start_dim=1)
        x = self._droput(torch.relu(self._bnorm4(self._aff1(x))))
        x = self._droput(torch.relu(self._bnorm5(self._aff2(x))))
        x = self._aff3(x)
        return x 

    def __str__(self):
        return 'ChessClassifierCNN'


class ChessRegressorCNN(nn.Module):
    def __init__(self):
        super(ChessRegressorCNN, self).__init__()

    def forward(self, x):
        return x
    
    def __str__(self):
        return "ChessRegressorCNN"


class DatasetWrapper(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class DatasetChess:
    def __init__(self, shuffle=True, batch_size=4096, fname='datagen/2019-serialized_NP-C.npz'):
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
        X, Y = X[:20000], Y[:20000]
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
    model, device = ChessClassifierCNN(3), 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # condition = torch.nn.MSELoss()
    condition = torch.nn.CrossEntropyLoss()
    model.to(device)
    summary(model,(17, 8, 8))
    fit(model, dataset.dataloaders['train'], dataset.dataloaders['valid'], optimizer, condition, device)


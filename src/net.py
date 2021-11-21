"""
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

from .trainer import PyTrainer
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import class_weight
from .utils import WPRINT, EPRINT, TPRINT


def calc_class_weights(labels):
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    print(cw)

class ChessClassifierCNN(nn.Module):
    def __init__(self, n_classes):
        super(ChessClassifierCNN, self).__init__()
        self._forward = nn.Sequential(
                nn.Conv2d(18, 32, 5),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Flatten(start_dim=1),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, n_classes),
                nn.ELU()
                )

        self._forward1 = nn.Sequential(
                # N x 32 x 8 x 8
                nn.Conv2d(18, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, 3),
                nn.BatchNorm2d(32),
                nn.ELU(),
                # N x 64 x 6 x 6
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64),
                nn.ELU(),
                # N x 128 x 4 x 4
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ELU(),
                # N x 128 x 2 x 2
                nn.Flatten(start_dim=1),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, n_classes),
                nn.ELU()
                )

    def forward(self, x):
        x = self._forward1(x)
        return x 

    def __str__(self):
        return 'ChessClassifierCNN'


class ChessRegressorCNN(nn.Module):
    def __init__(self):
        super(ChessRegressorCNN, self).__init__()
        self._forward = nn.Sequential(
                nn.Conv2d(18, 32, 5),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Flatten(start_dim=1),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 1),
                nn.Tanh()
                )

        self._forward1 = nn.Sequential(
                # N x 32 x 8 x 8
                nn.Conv2d(18, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, 3),
                nn.BatchNorm2d(32),
                nn.ELU(),
                # N x 64 x 6 x 6
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, 3),
                nn.BatchNorm2d(64),
                nn.ELU(),
                # N x 128 x 4 x 4
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, 3),
                nn.BatchNorm2d(128),
                nn.ELU(),
                # N x 128 x 2 x 2
                nn.Flatten(start_dim=1),
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 1),
                nn.Tanh()
                )


    def forward(self, x):
        x = self._forward1(x)
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
    def __init__(self, shuffle=True, batch_size=4096, fname='../data/2019-serialized-scaled_NP-R.npz'):
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
        WPRINT("loaded {} samples from {}".format(X.shape, fname), str(self), True)
        return X, Y

    def _train_valid_test_split(self, p=0.6):
        WPRINT("splitting data into train/valid/test", str(self), True)
        indices, n_samples = np.arange(self.X.shape[0]), int(4000000*p)
        np.random.shuffle(indices)
        X, Y = self.X[indices], self.Y[indices]
        X, Y = X[:4000000], Y[:4000000]
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
    model, device = ChessRegressorCNN(), 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    #model.load_state_dict(torch.load('../models/ChessClassifierCNN_subset_C-9.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #condition = nn.CrossEntropyLoss()
    condition = nn.MSELoss()
    problem = 'R'
    summary(model,(18, 8, 8))
    trainer = PyTrainer(model, 
                        device, 
                        problem=problem,
                        train=dataset.dataloaders['train'],
                        valid=dataset.dataloaders['valid'],
                        test=dataset.dataloaders['test'],
                        optimizer=optimizer,
                        condition=condition,
                        n_epochs=40,
                        verbose=True)
    trainer.fit()
    #trainer.test_classification()
    trainer.test_regression()
    #data = np.load('model-history.npz', allow_pickle=True)['arr_0'].item()
    #trainer.plot_classification(data=data)


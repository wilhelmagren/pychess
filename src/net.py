import os
import time
import chess
import torch
import numpy as np
np.random.seed(98)

from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from utils import WPRINT, EPRINT, TPRINT




class DeepMLP(torch.nn.Module):
    def __init__(self):
        super(DeepMLP, self).__init__()

        self._affine_layers = torch.nn.Sequential(
                torch.nn.Linear(13*8*8, 512),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(512, 128),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(128, 1),
                torch.nn.Dropout(p=0.5)
                )

    def forward(self, x):
        x = x.view(-1, 13*8*8)
        x = self._affine_layers(x)
        return x


class RegressNet(torch.nn.Module):
    def __init__(self):
        super(RegressNet, self).__init__()
    
        self._convolutional_layers = torch.torch.nn.Sequential(
                torch.torch.nn.Conv2d(in_channels=13, out_channels=8, kernel_size=(4,4)),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(4,4)),
                torch.nn.BatchNorm2d(8)
                )

        self._affine_layers = torch.nn.Sequential(
                torch.nn.Linear(8*2*2, 4*2*2),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4*2*2, 1),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.5)
                )

    def forward(self, x):
        x = self._convolutional_layers(x)
        x = x.view(-1, 8*2*2)
        x = self._affine_layers(x)
        return x


class DatasetWrapper(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class DatasetChess:
    def __init__(self, shuffle=True, batch_size=1024, fname='datagen/2019-scaled-serialized_FEN-R_SMALL.npz'):
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
    dset = DatasetChess()
    model, device = DeepMLP(), 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters())
    condition = torch.nn.MSELoss()
    summary(model,(1, 13, 8, 8))
    for epoch in range(10): 
        avg_tloss = 0
        for batch, (sample, label) in enumerate(dset.dataloaders['train']):
            sample, label = sample.to(device).float(), label.to(device).float()
            output = model(sample)
            loss = condition(output, label)
            avg_tloss += loss.item()
            loss.backward()
            optimizer.step()
        TPRINT(epoch, avg_tloss)



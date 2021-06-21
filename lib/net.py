"""
Author: Wilhelm Ågren
Last edited: 16/06-2021

Runs with CUDA 11.0 using #!usr/bin/python, not python3!!!!
"""


import os
import time
import chess
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
plt.style.use('ggplot')


class ChessClassifier(nn.Module):
    def __init__(self):
        super(ChessClassifier, self).__init__()
        self.l1 = nn.Linear(in_features=19*8*8, out_features=1024)
        self.l2 = nn.Linear(in_features=1024, out_features=512)
        self.l3 = nn.Linear(in_features=512, out_features=256)
        self.l4 = nn.Linear(in_features=256, out_features=3)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, 19*8*8)
        x = self.dropout(F.relu(self.l1(x)))
        x = self.dropout(F.relu(self.l2(x)))
        x = self.dropout(F.relu(self.l3(x)))
        x = F.relu(self.l4(x))
        return x


class Data(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]


class DataGen:
    def __init__(self, categorical=False):
        X, Y = self.dataloader(categorical)
        self.datasets = self.split(X, Y, 0.2)
        # self.histogram()

    def histogram(self):
        plt.hist(self.datasets['train'].Y, bins=3, color='gray', edgecolor='black', linewidth='1.2')
        plt.title('Categorical distribution')
        plt.xlabel('black-      draw      white+')
        plt.ylabel('num training samples')
        plt.xlim((-1, 3))
        plt.ylim((0, 1.3*self.datasets['train'].Y.shape[0]/3))
        plt.show()
        plt.hist(self.datasets['valid'].Y, bins=3, color='gray', edgecolor='black', linewidth='1.2')
        plt.title('Categorical distribution')
        plt.xlabel('black-      draw      white+')
        plt.ylabel('num validation samples')
        plt.xlim((-1, 3))
        plt.ylim((0, 1.3*self.datasets['valid'].Y.shape[0]/3))
        plt.show()

    @staticmethod
    def dataloader(categorical):
        X, Y = [], []
        for file in os.listdir('../parsed/'):
            if file.__contains__('{}_BIG'.format('_C' if categorical else '_R')):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file), allow_pickle=True)
                X.append(data['arr_0'])
                Y.append(data['arr_1'])

        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        print(' | loaded {} samples'.format(X.shape))
        return X, Y

    @staticmethod
    def split(X, Y, percentage):
        indices, splitamnt = np.arange(X.shape[0]), X.shape[0]*percentage
        np.random.shuffle(indices)
        X, Y = X[indices, :, :], Y[indices]
        x_train, x_valid = X[int(splitamnt):], X[:int(splitamnt)]
        y_train, y_valid = Y[int(splitamnt):], Y[:int(splitamnt)]
        print(' | split data into {:.0f}/{:.0f},  {}  {}'.format(100*(1 - percentage), 100*percentage, x_train.shape, x_valid.shape))
        datasets = {'train': Data(x_train, y_train), 'valid': Data(x_valid, y_valid)}
        return datasets


def plothistory(t, v):
    plt.plot([x for x in range(len(t))], t, color='maroon', label='training loss')
    plt.plot([x for x in range(len(v))], v, color='navy', label='validation loss')
    plt.title('loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.system('clear')
    # Train the net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' | training on device {}'.format(device))
    data = DataGen(categorical=True)
    model = ChessClassifier()
    model.to(device)
    summary(model, (1, 1216))
    optimizer, floss = optim.Adam(model.parameters(), lr=1e-4), nn.CrossEntropyLoss()
    traindata = DataLoader(data.datasets['train'], batch_size=8192, shuffle=True)
    validdata = DataLoader(data.datasets['valid'], batch_size=4096, shuffle=False)
    model.train()
    starttime, thist, vhist = time.time(), [], []
    for epoch in range(50):
        tloss, vloss = 0, 0
        for batch_idx, (sample, label) in enumerate(traindata):
            # Performing backprop after each training batch
            sample, label = sample.to(device).float(), label.to(device).long()
            optimizer.zero_grad()
            output = model(sample)
            loss = floss(output, label)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        # Don't perform any gradient steps, only count loss
        for batch_idx, (sample, label) in enumerate(validdata):
            sample, label = sample.to(device).float(), label.to(device).long()
            output = model(sample)
            loss = floss(output, label)
            vloss += loss.item()
        thist.append(tloss / len(traindata))
        vhist.append(vloss / len(validdata))
        print(' | epoch\t{},\ttloss\t{:.3f}\tvloss\t{:.3f}'.format(epoch + 1, tloss / len(traindata), vloss / len(validdata)))
    print('-|' + '-' * 62)
    print(' | done! Training took {:.0f}s'.format(time.time() - starttime))
    torch.save(model.state_dict(), "../nets/ChessClassifier.pth")
    plothistory(thist, vhist)

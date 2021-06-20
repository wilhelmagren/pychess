"""
Author: Wilhelm Ågren
Last edited: 16/06-2021

Runs with CUDA 11.0 using #!usr/bin/python, not python3!!!!
"""


import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
plt.style.context('classic')


class ChessNet(nn.Module):
    """
    ------------- ChessNet --------------
    1st layer: 11x8x8 => 16x6x6
    2nd layer: 16x6x6 => 32x4x4
    3rd layer: 32x4x4 => 64x2x2
    4th layer: 64x2x2 => 128
    5th layer: 128 => 128
    6th layer: 128 => 64
    7th layer: 64 => 1
    -------------------------------------
    """
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.affine1 = nn.Linear(in_features=512, out_features=256)
        self.affine2 = nn.Linear(in_features=256, out_features=128)
        self.affine3 = nn.Linear(in_features=128, out_features=1)

        # Regularization
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=16)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)

    def forward(self, x) -> torch.tensor:
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = x.view(-1, 512)
        x = F.relu(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        x = self.affine3(x)
        return x


class PartitionDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> np.array:
        return self.X[idx], self.Y[idx]


class ChessDataset(object):
    def __init__(self):
        X, Y = self.dataloader()
        self.histogram(Y)
        self.datasets = self.split(X, Y, 0.2)

    @staticmethod
    def split(X, Y, split) -> dict:
        train_x, tmp_x, train_y, tmp_y = train_test_split(X, Y, test_size=split)
        valid_x, test_x, valid_y, test_y = train_test_split(tmp_x, tmp_y, test_size=0.5)
        print(' | split:  train({}, )  valid({}, )  test({}, )'.format(train_x.shape[0], valid_x.shape[0], test_x.shape[0]))
        return {'train': PartitionDataset(train_x, train_y),
                'valid': PartitionDataset(valid_x, valid_y),
                'test': PartitionDataset(test_x, test_y)}

    @staticmethod
    def dataloader() -> (np.array, np.array):
        X, Y = [], []
        for file in os.listdir('../parsed/'):
            if file.__contains__('dataset01_BIG'):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file))
                x, y = data['arr_0'], data['arr_1']
                X.append(x)
                Y.append(y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        print(' | loaded ({}, ) samples'.format(X.shape[0]))
        return X, Y

    @staticmethod
    def normalize(d) -> (np.array, np.array):
        """ Min-max feature scaling, with complementary (a, b)
        Y = (d[1] + 30)/60
        Y = -1 + 2*Y
        """
        Y = d[1]
        Y[Y > 10] = 10
        Y[Y < -10] = -10
        return d[0], Y

    @staticmethod
    def histogram(x) -> None:
        plt.hist(x, bins=40, color='maroon')
        plt.xlabel('state evaluation')
        plt.ylabel('num samples')
        plt.show()


def fit(net, opti, floss, traindata, validdata, epochs, dev) -> (list, list):
    starttime, thistory, vhistory = time.time(), [], []
    model.train()
    for epoch in range(epochs):
        tloss, numtloss = 0, 0
        vloss, numvloss = 0, 0
        for batch_idx, (d, t) in enumerate(traindata):
            d, t = d.to(dev).float(), torch.unsqueeze(t.to(dev), -1).float()
            opti.zero_grad()
            output = net(d)
            loss = floss(output, t)
            loss.backward()
            opti.step()
            tloss += loss.item()
            numtloss += 1
        with torch.no_grad():
            for batch_idx, (d, t) in enumerate(validdata):
                d, t = d.to(dev).float(), torch.unsqueeze(t.to(dev), -1).float()
                output = net(d)
                loss = floss(output, t)
                vloss += loss.item()
                numvloss += 1
        thistory.append(tloss/numtloss)
        vhistory.append(vloss/numvloss)
        print(' | epoch\t{},\ttloss\t{:.3f}\tvloss\t{:.3f}'.format(epoch + 1, tloss/numtloss, vloss/numvloss))
    print('-|'+'-'*62)
    print(' | done! Training took {:.0f}s'.format(time.time() - starttime))
    torch.save(net.state_dict(), "../nets/ChessNet.pth")

    return thistory, vhistory


def validate(net, floss, testdata, dev) -> None:
    tloss, tnumloss = 0, 0
    for batch_idx, (d, t) in enumerate(testdata):
        d, t = d.to(dev).float(), torch.unsqueeze(t.to(dev), -1).float()
        output = net(d)
        loss = floss(output, t)
        tloss += loss.item()
        tnumloss += 1
    print(' | final testing result,\tL1 loss {:.3f}'.format(tloss/tnumloss))


def plothistory(t, v):
    plt.plot([x for x in range(len(t))], t, color='maroon', label='training loss')
    plt.plot([x for x in range(len(v))], v, color='navy', label='validation loss')
    plt.title('loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()


def writehist(t, v):
    with open('../history.txt', 'a') as f:
        for (tval, vval) in zip(t, v):
            f.write('{},{}\n'.format(tval, vval))


if __name__ == '__main__':
    os.system('clear')
    # Train the net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FULLDATA = ChessDataset()
    train_dataloader = DataLoader(FULLDATA.datasets['train'], batch_size=8192, shuffle=True)
    valid_dataloader = DataLoader(FULLDATA.datasets['valid'], batch_size=1024, shuffle=False)
    test_dataloader = DataLoader(FULLDATA.datasets['test'], batch_size=1024, shuffle=False)
    model = ChessNet()
    # model.load_state_dict(torch.load('../nets/ChessNet.pth', map_location=lambda storage, loc: storage))
    model.to(device)
    optimizer, ffloss = optim.Adam(model.parameters(), lr=1e-4), nn.L1Loss()
    summary(model, (11, 8, 8))
    this, vhis = fit(model, optimizer, ffloss, train_dataloader, valid_dataloader, 10, device)
    validate(model, ffloss, test_dataloader, device)
    plothistory(this, vhis)
    writehist(this, vhis)

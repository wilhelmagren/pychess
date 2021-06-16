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
import torch.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=16, kernel_size=(5, 5))  # in: 11x8x8, out: 16x4x4
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))  # in: 16x4x4, out: 32x2x2
        self.aff1 = nn.Linear(in_features=128, out_features=128)
        self.aff2 = nn.Linear(in_features=128, out_features=64)
        self.aff3 = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.view(-1, 128)
        x = torch.tanh(self.aff1(x))
        x = torch.tanh(self.aff2(x))
        x = self.aff3(x)
        return x


class ChessDataset(Dataset):
    def __init__(self):
        self.X, self.Y = self.dataloader()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> (np.array, np.array):
        return self.X[idx], self.Y[idx]

    @staticmethod
    def dataloader() -> (np.array, np.array):
        X, Y = [], []
        for file in os.listdir('../parsed/'):
            if file.__contains__('_R'):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file))
                x, y = data['arr_0'], data['arr_1']
                X.append(x)
                Y.append(y)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        return X, Y


def train(net, opti, floss, data, epochs, dev):
    starttime = time.time()
    for epoch in range(epochs):
        all_loss = 0
        num_loss = 0
        for batch_idx, (d, t) in enumerate(data):
            d, t = d.to(dev).float(), torch.unsqueeze(t.to(dev), -1).float()
            opti.zero_grad()
            output = net(d)
            loss = floss(output, t)
            loss.backward()
            opti.step()
            all_loss += loss.item()
            num_loss += 1
        print(' | epoch\t{},\tloss\t{:.2f}'.format(epoch + 1, all_loss/num_loss))
    print('-|'+'-'*63)
    print(' | done! Training took {:.2f}s'.format(time.time() - starttime))


if __name__ == '__main__':
    # Train the net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(ChessDataset(), batch_size=1024, shuffle=True)
    model = ChessNet()
    model.to(device)
    optimizer, ffloss = optim.Adam(model.parameters()), nn.L1Loss()
    summary(model, (11, 8, 8))
    model.train()
    train(model, optimizer, ffloss, dataloader, 10, device)

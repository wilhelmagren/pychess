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
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=11, out_channels=16, kernel_size=(5, 5), padding=(1, 1))  # in: 11x8x8, out: 16x6x6
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))  # in: 16x6x6, out: 32x4x4
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # in: 32x4x4, out: 64x2x2
        self.aff1 = nn.Linear(in_features=256, out_features=128)
        self.aff2 = nn.Linear(in_features=128, out_features=128)
        self.aff3 = nn.Linear(in_features=128, out_features=64)
        self.aff4 = nn.Linear(in_features=64, out_features=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x) -> torch.tensor:
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, 256)
        x = torch.tanh(self.aff1(x))
        x = torch.tanh(self.aff2(x))
        x = torch.tanh(self.aff3(x))
        x = self.aff4(x)
        return x


class ChessDataset(Dataset):
    def __init__(self):
        self.X, self.Y = self.dataloader()
        self.histogram(self.Y)

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
        print(' | loaded {} samples'.format(X.shape[0]))
        return X, Y

    @staticmethod
    def histogram(x) -> None:
        plt.hist(x, bins=60, color='maroon')
        plt.xlabel('state evaluation')
        plt.ylabel('num samples')
        plt.show()


def train(net, opti, floss, data, epochs, dev) -> list:
    starttime, history = time.time(), []
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
        history.append(all_loss/num_loss)
        print(' | epoch\t{},\tloss\t{:.2f}'.format(epoch + 1, all_loss/num_loss))
    print('-|'+'-'*62)
    print(' | done! Training took {:.0f}s'.format(time.time() - starttime))
    torch.save(net.state_dict(), "../nets/ChessNet.pth")
    return history


def plothistory(history):
    plt.plot([x for x in range(len(history))], history, color='maroon', label='training loss')
    plt.title('loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('L1 loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    os.system('clear')
    # Train the net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(ChessDataset(), batch_size=16384, shuffle=True)
    model = ChessNet()
    model.to(device)
    optimizer, ffloss = optim.Adam(model.parameters()), nn.L1Loss()
    summary(model, (11, 8, 8))
    model.train()
    his = train(model, optimizer, ffloss, dataloader, 30, device)
    plothistory(his)

"""
Author: Wilhelm Ågren
Last edited: 16/06-2021

Runs with CUDA 11.0 using #!usr/bin/python, not python3!!!!

NOTES:
    When training AutoEncoder - use MSE for criterion. MSE requires data.float().
    When training classifier  - use NLLLoss or CrossEntropyLoss, requires data.long().
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


class ChessAE(nn.Module):
    def __init__(self):
        super(ChessAE, self).__init__()
        self.encoder1 = nn.Linear(in_features=775, out_features=512)
        self.encoder2 = nn.Linear(in_features=512, out_features=256)
        self.encoder3 = nn.Linear(in_features=256, out_features=128)
        self.decoder1 = nn.Linear(in_features=128, out_features=256)
        self.decoder2 = nn.Linear(in_features=256, out_features=512)
        self.decoder3 = nn.Linear(in_features=512, out_features=775)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, 775)
        x = self.dropout(torch.tanh(self.encoder1(x)))
        x = self.dropout(torch.tanh(self.encoder2(x)))
        x = self.dropout(torch.tanh(self.encoder3(x)))
        x = self.dropout(torch.tanh(self.decoder1(x)))
        x = self.dropout(torch.tanh(self.decoder2(x)))
        x = self.dropout(torch.tanh(self.decoder3(x)))
        return x


class ChessClassifier(nn.Module):
    def __init__(self):
        """
        MLP Architecture:
        [affine-BN-ReLU-dropout]x3-affine-softmax
        """
        super(ChessClassifier, self).__init__()
        self.l1 = nn.Linear(in_features=391, out_features=1024)
        self.l2 = nn.Linear(in_features=1024, out_features=512)
        self.l3 = nn.Linear(in_features=512, out_features=256)
        self.l4 = nn.Linear(in_features=256, out_features=3)
        self.dropout = nn.Dropout(p=0.25)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = x.view(-1, 6*64 + 7)
        x = self.dropout(torch.tanh(self.bn1(self.l1(x))))
        x = self.dropout(torch.tanh(self.bn2(self.l2(x))))
        x = self.dropout(torch.tanh(self.bn3(self.l3(x))))
        x = self.l4(x)
        return x


class CNNChessClassifier(nn.Module):
    # Not special with flags
    def __init__(self):
        super(CNNChessClassifier, self).__init__()
        self.c1 = nn.Conv2d(in_channels=13, out_channels=16, kernel_size=(3, 3))
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.c3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.dropout = nn.Dropout(p=0.25)
        # Since batch normalization on C dimension, commonly refered to as 'Spatial Batch Normalization'
        self.bn1 = nn.BatchNorm2d(16, 6, 6)
        self.bn2 = nn.BatchNorm2d(32, 4, 4)
        self.bn3 = nn.BatchNorm2d(64, 2, 2)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.c1(x)))
        x = torch.tanh(self.bn2(self.c2(x)))
        x = torch.tanh(self.bn3(self.c3(x)))
        # 64*2*2 = 256
        x = x.view(-1, 256)
        x = self.dropout(torch.tanh(self.bn4(self.l1(x))))
        x = self.dropout(torch.tanh(self.bn5(self.l2(x))))
        x = self.dropout(torch.tanh(self.bn6(self.l3(x))))
        x = self.l4(x)  # ????? negative log-likelihood does the softmax activation fuynction for me. so...

        return x


class CNNChessClassifierMod(nn.Module):
    def __init__(self):
        super(CNNChessClassifierMod, self).__init__()
        self.c1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3))
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.c3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.dropout = nn.Dropout(p=0.25)
        # Since batch normalization on C dimension, commonly refered to as 'Spatial Batch Normalization'
        self.bn1 = nn.BatchNorm2d(16, 6, 6)
        self.bn2 = nn.BatchNorm2d(32, 4, 4)
        self.bn3 = nn.BatchNorm2d(64, 2, 2)
        self.l1 = nn.Linear(256+7, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 3)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

    def forward(self, x):
        # shape(x): Bx13x8x8
        conv_data = x[:, :6, :, :]
        bitmap_flags = x[:, 6:, 0, 0]
        x = torch.tanh(self.bn1(self.c1(conv_data)))
        x = torch.tanh(self.bn2(self.c2(x)))
        x = torch.tanh(self.bn3(self.c3(x)))
        # 64*2*2 = 256
        x = x.view(-1, 256)
        x = torch.cat((x, bitmap_flags), dim=1)
        x = self.dropout(torch.tanh(self.bn4(self.l1(x))))
        x = self.dropout(torch.tanh(self.bn5(self.l2(x))))
        x = self.dropout(torch.tanh(self.bn6(self.l3(x))))
        x = self.l4(x)
        return x


class ChessRegression(nn.Module):
    def __init__(self):
        """
        MLP Architecture:
        [affine-BN-ReLU-dropout]x3-affine-softmax
        """
        super(ChessRegression, self).__init__()
        self.l1 = nn.Linear(in_features=391, out_features=1024)
        self.l2 = nn.Linear(in_features=1024, out_features=512)
        self.l3 = nn.Linear(in_features=512, out_features=256)
        self.l4 = nn.Linear(in_features=256, out_features=1)
        self.dropout = nn.Dropout(p=0.25)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.bn3 = nn.BatchNorm1d(num_features=256)

    def forward(self, x):
        x = x.view(-1, 6*64 + 7)
        x = self.dropout(torch.tanh(self.bn1(self.l1(x))))
        x = self.dropout(torch.tanh(self.bn2(self.l2(x))))
        x = self.dropout(torch.tanh(self.bn3(self.l3(x))))
        x = self.l4(x)
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
        self.datasets = self.split(X, Y, 0.1)  #
        self.histogram()

    def histogram(self):
        plt.hist(self.datasets['train'].Y, bins=3, color='gray', edgecolor='black', linewidth='1.2')
        plt.title('Categorical distribution')
        plt.xlabel('black-      draw      white+')
        plt.ylabel('num training samples')
        plt.xlim((-1, 3))
        # plt.ylim((0, 1.3*self.datasets['train'].Y.shape[0]/3))
        plt.show()
        plt.hist(self.datasets['valid'].Y, bins=3, color='gray', edgecolor='black', linewidth='1.2')
        plt.title('Categorical distribution')
        plt.xlabel('black-      draw      white+')
        plt.ylabel('num validation samples')
        plt.xlim((-1, 3))
        # plt.ylim((0, 1.3*self.datasets['valid'].Y.shape[0]/3))
        plt.show()

    @staticmethod
    def dataloader(categorical):
        X, Y = [], []
        for file in os.listdir('../parsed/'):
            if file.__contains__('TEST'):
                continue
            if file.__contains__('dense_CNN_TRAIN'.format('C' if categorical else 'R')):
                print(' | parsing data from filepath {}'.format(file))
                data = np.load(os.path.join('../parsed/', file), allow_pickle=True)
                X.append(data['arr_0'])
                Y.append(data['arr_1'])

        X, Y = np.concatenate(X, axis=0), np.concatenate(Y, axis=0)
        # X = np.concatenate(X, axis=0)
        print(' | loaded {} samples'.format(X.shape))
        return X, Y

    @staticmethod
    def split(X, Y, percentage):
        indices, splitamnt = np.arange(X.shape[0]), X.shape[0]*percentage
        np.random.shuffle(indices)
        X, Y = X[indices, :], Y[indices]
        # Y = 2*(Y + 15000)/30000
        x_train, x_valid = X[int(splitamnt):], X[:int(splitamnt)]
        y_train, y_valid = Y[int(splitamnt):], Y[:int(splitamnt)]
        print(' | split data into {:.0f}/{:.0f},  {}  {}'.format(100*(1 - percentage), 100*percentage, x_train.shape, x_valid.shape))
        datasets = {'train': Data(x_train, y_train), 'valid': Data(x_valid, y_valid)}
        return datasets


def plothistory(tl, vl, ta, va):
    plt.plot([x for x in range(len(tl))], tl, color='royalblue', label='training', linewidth=1.2)
    plt.plot([x for x in range(len(vl))], vl, color='forestgreen', label='validation', linewidth=1.2)
    plt.title('loss evolution')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.show()

    plt.plot([x for x in range(len(ta))], ta, color='royalblue', label='training', linewidth=1.2)
    plt.plot([x for x in range(len(va))], va, color='forestgreen', label='validation', linewidth=1.2)
    plt.title('accuracy evolution')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()



def test(model, testdata):
    acc = 0
    model.load_state_dict(torch.load('../nets/denseClassifier.pth', map_location=lambda storage, loc: storage))
    for batch_idx, (sample, label) in enumerate(testdata):
        sample, label = sample.to(device).float(), label.to(device).long()
        output = model(sample)
        _, predicted = torch.max(output.data, 1)
        acc += (predicted == label).sum().item() / output.shape[0]
    print(' | testing accuracy: {:.1f}%'.format(100*acc/len(testdata)))


if __name__ == '__main__':
    os.system('clear')
    # Train the net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' | training on device {}'.format(device))
    data = DataGen(categorical=True)
    model = CNNChessClassifierMod()
    model.to(device)
    summary(model, (13, 8, 8))
    optimizer, floss = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4), nn.CrossEntropyLoss()
    #testdata = DataLoader(data.datasets['test'], batch_size=4096, shuffle=True)
    #"""
    traindata = DataLoader(data.datasets['train'], batch_size=8192, shuffle=True)
    validdata = DataLoader(data.datasets['valid'], batch_size=4096, shuffle=False)
    model.train()
    starttime, tlhist, tahist, vlhist, vahist = time.time(), [], [], [], []
    for epoch in range(50):
        tloss, tacc, vloss, vacc = 0, 0, 0, 0
        for batch_idx, (sample, label) in enumerate(traindata):
            # Performing backprop after each training batch
            sample, label = sample.to(device).float(), label.to(device).long()
            optimizer.zero_grad()
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            tacc += (predicted == label).sum().item()/output.shape[0]
            loss = floss(output, label)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        # Don't perform any gradient steps, only count loss
        for batch_idx, (sample, label) in enumerate(validdata):
            sample, label = sample.to(device).float(), label.to(device).long()
            output = model(sample)
            _, predicted = torch.max(output.data, 1)
            vacc += (predicted == label).sum().item()/output.shape[0]
            loss = floss(output, label)
            vloss += loss.item()
        tlhist.append(tloss / len(traindata))
        tahist.append(tacc / len(traindata))
        vlhist.append(vloss / len(validdata))
        vahist.append(vacc / len(validdata))
        print(' | ep {:02d},  tloss {:.4f}   vloss {:.4f},  tacc {:.1f}%   vacc {:.1f}%'.format(epoch + 1, tloss / len(traindata), vloss / len(validdata), 100*tacc / len(traindata), 100*vacc / len(validdata)))
    print('-|' + '-' * 62)
    print(' | done! Training took {:.0f}s'.format(time.time() - starttime))
    torch.save(model.state_dict(), "../nets/denseCNNClassifierMod.pth")
    plothistory(tlhist, vlhist, tahist, vahist)
    #"""
    #test(model, testdata)

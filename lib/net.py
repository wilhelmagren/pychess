"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 30/05-2021
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
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SmallDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ChessDataset(Dataset):
    """
    Load the parsed dataset for regression positions, i.e. targets are 'Stockfish 13' evaluation.
    Inherits from pytorch data utility 'Dataset' to create iterative set of data samples for Net.

    |   Featuring class methods:
    |
    |   func __init__/2
    |   @spec  ::  (Dataset(), str) => (Class(ChessDataset))
    |
    |   func __len__/1
    |   @spec  ::  (Class()) => (int)
    |
    |   func __getitem__/2
    |   @spec  ::  (Class(), int) => (np.array, np.array)
    |
    |   func __read__/1
    |   @spec  ::  (str) => (np.array, np.array)

    Expects the loaded data to be of shape (num_samples, 7, 8, 8) given by the State serialization.
    Correspondingly the targets, Y, has to have same first shape as data samples, i.e. X.shape[0] == Y.shape[0]
    """
    def __init__(self, datadir='../parsed/', ptype='_R', split=0.3):
        self.X, self.Y = self.__read__(datadir, ptype)
        self.__visualize__()
        self.datasets = {}
        if ptype != '_C':
            self.__normalize__(full=False, a=-1, b=1)
            self.__visualize__()
        else:
            train_x, train_y, valid_x, valid_y, test_x, test_y = self.__split__(split)
            self.datasets = {'train': SmallDataset(train_x, train_y),
                             'valid': SmallDataset(valid_x, valid_y),
                             'test': SmallDataset(test_x, test_y)}

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> (np.array, np.array):
        return self.X[idx], self.Y[idx]

    def __split__(self, split) -> (np.array, np.array, np.array, np.array, np.array, np.array):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices, :, :, :]
        self.Y = self.Y[indices]
        train_x, tmp_x, train_y, tmp_y = train_test_split(self.X, self.Y, test_size=split)
        valid_x, test_x, valid_y, test_y = train_test_split(tmp_x, tmp_y, test_size=0.5)
        print(f' | loaded training\t{train_x.shape},\t{train_y.shape}\n'
              f' | loaded valid\t\t{valid_x.shape},\t{valid_y.shape}\n'
              f' | loaded testing\t{test_x.shape},\t{test_y.shape}')
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def __normalize__(self, full=False, a=0, b=1):
        """
        Min-max feature scaling, brings all values into the range [0, 1]. Also called unity-based normalization.
        Can be used to restrict the range of values between any arbitrary points a, b.
        X' = a + (X- Xmin)(b - a)/(Xmax - Xmin)
        """
        self.Y[self.Y > 15] = 15
        self.Y[self.Y < -15] = -15
        if full:
            self.Y = (self.Y + 15)/30
            self.Y = a + self.Y*(b - a)

    def __visualize__(self):
        plt.hist(self.Y, bins=30, color='maroon')
        plt.xlabel('target evaluation')
        plt.ylabel('num labels')
        plt.show()

    @staticmethod
    def __read__(datadir, ptype):
        x, y = [], []
        for file in os.listdir(datadir):
            if file.__contains__(ptype):
                print(' | parsing data from filepath, {}'.format(file))
                data = np.load(os.path.join('../parsed/', file))
                X, Y = data['arr_0'], data['arr_1']
                x.append(np.array(X, dtype=np.int32))
                y.append(np.array(Y, dtype=np.int32))
        X = np.concatenate(x, axis=0)
        Y = np.concatenate(y, axis=0)

        assert X.shape == (X.shape[0], 7, 8, 8)
        assert Y.shape == (Y.shape[0], )
        print(f' | loaded total\t\t{X.shape},\t{Y.shape}')
        return X, Y


class TinyPruneNet(nn.Module):
    def __init__(self):
        super(TinyPruneNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(5, 5))  # in: 7x8x8, out: 32x4x4
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512)  # 32x4x4
        x = torch.tanh((self.fc1(x)))
        x = self.fc2(x)
        return x


class TinyChessNet(nn.Module):
    def __init__(self):
        super(TinyChessNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=(5, 5))  # in: 7x8x8, out: 32x4x4
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 512)  # 32x4x4
        x = torch.tanh((self.fc1(x)))
        x = self.fc2(x)
        return x


def fit_classifier(model, device, traindata, validdata, optimizer, floss, num_epochs, batch_size):
    model.train()
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    for epoch in range(num_epochs):
        avg_tloss, avg_tacc, avg_vloss, avg_vacc = 0, 0, 0, 0
        for batch_idx, (data, target) in enumerate(traindata):
            data, target = data.to(device).float(), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = floss(output, target)
            _, predicted = torch.max(output.data, 1)
            avg_tacc += (predicted == target).sum().item()/batch_size
            avg_tloss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(' | Training epoch: {}\t[({:.1f}%)]\tloss: {:.6f}'.format(epoch + 1, 100.*batch_idx/len(traindata), loss.item()))
        # Calculate validation loss & acc after each epoch
        for vbatch_idx, (vdata, vtarget) in enumerate(validdata):
            data, target = vdata.to(device).float(), vtarget.to(device).long()
            output = model(data)
            loss = floss(output, target)
            _, predicted = torch.max(output.data, 1)
            avg_vacc += (predicted == target).sum().item()/batch_size
            avg_vloss += loss.item()

        avg_tloss /= len(traindata)
        avg_tacc /= len(traindata)
        avg_vloss /= len(validdata)
        avg_vacc /= len(validdata)
        train_loss.append(avg_tloss)
        train_acc.append(avg_tacc)
        valid_loss.append(avg_vloss)
        valid_acc.append(avg_vacc)
        print(' | Average loss: {:.5f},\taverage acc: V={:.1f}%\n'.format(avg_vloss, 100. * avg_vacc))
        torch.save(model.state_dict(), "../nets/tiny_class_value.pth")
        time.sleep(2)
    return [train_loss, train_acc, valid_loss, valid_acc]


def evaluate_classifier(model, device, dataset, batch_size):
    pass


def fit_regression(model, device, traindata, validdata, optimizer, floss, num_epochs, batch_size):
    model.train()
    train_loss, valid_loss = [], []
    for epoch in range(num_epochs):
        avg_tloss, avg_tacc, avg_vloss, avg_vacc = 0, 0, 0, 0
        for batch_idx, (data, target) in enumerate(traindata):
            data, target = data.to(device).float(), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = floss(output, target)
            avg_tloss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(' | Training epoch: {}\t[({:.1f}%)]\tloss: {:.6f}'.format(epoch + 1,
                                                                                100. * batch_idx / len(traindata),
                                                                                loss.item()))
        # Calculate validation loss & acc after each epoch
        for vbatch_idx, (vdata, vtarget) in enumerate(validdata):
            data, target = vdata.to(device).float(), vtarget.to(device).long()
            output = model(data)
            loss = floss(output, target)
            avg_vloss += loss.item()

        avg_tloss /= len(traindata)
        avg_vloss /= len(validdata)
        train_loss.append(avg_tloss)
        valid_loss.append(avg_vloss)
        print(
            ' | Average loss: [T={:.5f}, V={:.5f}]\n'.format(avg_tloss, avg_vloss))
        torch.save(model.state_dict(), "../nets/tiny_regr_value.pth")
        time.sleep(2)
    return [train_loss, valid_loss]


def plot(train, valid, label):
    plt.plot([x for x in range(1, len(train) + 1)], train, color='maroon', label='training {}'.format(label))
    plt.plot([x for x in range(1, len(valid) + 1)], valid, color='navy', label='validation {}'.format(label))
    plt.title('{} evolution'.format(label))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = TinyPruneNet()
    model.cuda()
    summary(model, (7, 8, 8))

    # Read and create the dataset structures
    chess_dataset = ChessDataset(ptype='_C')
    # train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=512, shuffle=True)
    optimizer = optim.Adagrad(model.parameters(), lr=0.025, eps=1e-8)
    #"""
    train_loader = torch.utils.data.DataLoader(chess_dataset.datasets['train'], batch_size=256, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(chess_dataset.datasets['valid'], batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(chess_dataset.datasets['test'], batch_size=256, shuffle=True)

    # Specify optimizer and loss function
    floss = nn.CrossEntropyLoss()
    l = fit_classifier(model, device, train_loader, valid_loader, optimizer, floss, 15, 512)
    # l = fit_regression(model, device, train_loader, valid_loader, optimizer, floss, 15, 512)
    plot(l[0], l[2], 'loss')




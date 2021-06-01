"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 30/05-2021
"""
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset


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
    |   @spec  ::  (class) => (int)
    |
    |   func __getitem__/2
    |   @spec  ::  (class, int) => (np.array, np.array)
    |
    |   func __read__/1
    |   @spec  ::  (str) => (np.array, np.array)

    Expects the loaded data to be of shape (num_samples, 7, 8, 8) given by the State serialization.
    Correspondingly the targets, Y, has to have same first shape as data samples, i.e. X.shape[0] == Y.shape[0]
    """
    def __init__(self, datadir='../parsed/'):
        self.X, self.Y = self.__read__(datadir)
        self.__visualize__()
        # self.__normalize__(a=-1, b=1)
        # self.__visualize__()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx) -> (np.array, np.array):
        return self.X[idx], self.Y[idx]

    def __normalize__(self, a=0, b=1):
        """
        Min-max feature scaling, brings all values into the range [0, 1]. Also called unity-based normalization.
        Can be used to restrict the range of values between any arbitrary points a, b.
        X' = a + (X- Xmin)(b - a)/(Xmax - Xmin)
        """
        self.Y[self.Y > 30] = 30
        self.Y[self.Y < -30] = -30
        self.Y = (self.Y + 30)/60
        self.Y = a + self.Y*(b - a)

    def __visualize__(self):
        plt.hist(self.Y, bins=60, color='maroon')
        plt.xlabel('target evaluation')
        plt.ylabel('num labels')
        plt.show()

    @staticmethod
    def __read__(datadir) -> (np.array, np.array):
        x, y = [], []
        for file in os.listdir(datadir):
            if file.__contains__('dataset_batch') or file.__contains__('dataset_tactics'):
                print(' | parsing data from filepath, {}'.format(file))
                data = np.load(os.path.join('../parsed/', file))
                X, Y = data['arr_0'], data['arr_1']
                x.append(X)
                y.append(Y)
        X = np.concatenate(x, axis=0)
        Y = np.concatenate(y, axis=0)

        assert X.shape == (X.shape[0], 7, 8, 8)
        assert Y.shape == (Y.shape[0], )
        print(f'loaded, {X.shape}, {Y.shape}')
        return X, Y


class TinyChessNet(nn.Module):
    """
    Build the Neural Net for evaluating chess positions. Train/Validate/Test on the ChessDataset object.
    Inherits from pytorch nn.module for layers composing the net. Also utilizes functional methods from
    torch.nn.functional to calculate activation functions.

    |   Featuring class methods:
    |   func __init__/1
    |   @spec  ::  (nn.Module) => (Class(TinyChessNet())
    |
    |   func __repr__/1
    |   @spec  ::  (class) => (str)
    |
    |   func forward/2
    |   @spec  ::  (class, torch.tensor) => (torch.tensor)

    This Neural Net is a smaller version of the original ChessNet. Instead of enhancing each DxD slice of the board
    three times it only performs this once. All in all it is a 6 layered CNN, with 4 initial convolutional layers
    followed by 2 affine layers. The convolutions are designed such that the representative 'size' of the board is
    halfed after each convolutional operation. Each convolutional layer is followed by the ReLU activation function,
    but the affine layer(s) use tanh, since the final output is expected to be in the range [-1, 1].
    TODO: Regularization technique - dropout - can be used if specified. Not fully tested for performance.
    """
    def __init__(self):
        super(TinyChessNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=8, kernel_size=(3, 3), padding=(1, 1))  # 8x8x8
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5))  # 16x4x4
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))  # 32x2x2
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2))  # 64x1x1
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.3)

    def __repr__(self):
        return 'TinyChessNet(nn.Module)'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 64)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chess_dataset = ChessDataset()
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=1024, shuffle=True)
    model = TinyChessNet()
    model.cuda()
    summary(model, (7, 8, 8))
    optimizer = optim.Adagrad(model.parameters(), lr=0.05, eps=1e-9)
    floss = nn.L1Loss()

    model.train()

    for epoch in range(40):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = torch.unsqueeze(target, -1)
            target = target.float()
            optimizer.zero_grad()
            output = model(data)
            loss = floss(output, target)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            num_loss += 1
        print("\nepoch %3d:  loss %f" % (epoch, all_loss / num_loss))
        torch.save(model.state_dict(), "../nets/tiny_value_big.pth")

"""
Author: Wilhelm Ågren, wagren@kth.se
Last edited: 28/05-2021
"""
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import os
import pandas as pd
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self):
        self.X, self.Y = self.read_files()
        print(f'loaded, {self.X.shape}, {self.Y.shape}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def read_files(self) -> (list, list):
        """
        func read_files/1
        @spec :: (Class(CNN)) => (list, list)
            Read all of the parsed_data found in '../parsed_data/' and extract
            the targets and data respectively. Returns the a list of the data and
            a list of the corresponding targets. Order of the elements are important! Can't be changed!!!
        """
        data = []
        column_list = []
        for x in range(7 * 8 * 8):
            column_list.append(f'x{x}')
        for file in os.listdir('../test_parsed/'):
            if '1000games' in file or '2000games' in file:
                print(f'<|\tParsing data from filepath :: ../parsed_data/{file}')
                data.append(pd.read_csv('../test_parsed/' + file))
        train_x = []
        train_y = []
        for dat in data:
            train_x.append(dat.loc[:, column_list])
            train_y.append(dat.loc[:, dat.columns == 'y'])
        x_data = np.concatenate(train_x, axis=0)
        y_data = np.concatenate(train_y, axis=0)
        x_data = np.reshape(x_data, (x_data.shape[0], 7, 8, 8))

        return x_data, y_data


class ChessDataset(Dataset):
    def __init__(self):
        data = np.load('../parsed/dataset_1C_R.npz')
        self.X = data['arr_0']
        self.Y = data['arr_1']
        print(f'loaded, {self.X.shape}, {self.Y.shape}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.a1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=1)  # 8x8x12 => 8x8x16
        self.a2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)  # 8x8x16 => 8x8x16
        self.a3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))  # 8x8x16 => 6x6x32

        self.b1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)  # 6x6x32 => 6x6x32
        self.b2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)  # 6x6x32 => 6x6x32
        self.b3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # 6x6x32 => 4x4x64

        self.c1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)  # 4x4x64 => 4x4x64
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)  # 4x4x64 => 4x4x64
        self.c3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))  # 4x4x64 => 2x2x128

        self.d1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)  # 2x2x128 => 2x2x128
        self.d2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)  # 2x2x128 => 2x2x128
        self.d3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2))  # 2x2x128 => 1x1x256

        self.e1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))  # 1x1x256
        self.e2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))  # 1x1x256
        self.e3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))  # 1x1x256

        self.last = nn.Linear(256, 1)

    def forward(self, x):
        # 8x8x16
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))
        # 6x6x32
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        # 4x4x64
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        # 2x2x128
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        # 1x1x256
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.e3(x))
        # 'Flatten'
        x = x.view(-1, 256)
        x = self.last(x)
        # Output
        return torch.tanh(x)


class resNet(nn.Module):
    def __init__(self):
        super(resNet, self).__init__()
        self.a1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3, 3), padding=1)  # 12x8x8 => 16x8x8
        self.a2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)  # 16x8x8 => 16x8x8
        self.a3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))  # 16x8x8 => 32x4x4

        self.b1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)  # 32x4x4 => 32x4x4
        self.b2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)  # 32x4x4 => 32x4x4
        self.b3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # 32x4x4 => 64x2x2

        self.c1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))  # 64x2x2 => 64x2x2
        self.c2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))  # 64x2x2 => 128x1x1

        self.d1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))  # 128x1x1

        self.last1 = nn.Linear(128, 128)
        self.last2 = nn.Linear(128, 64)
        self.last3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.tanh(self.a1(x))
        x = torch.tanh(self.a2(x))
        x = torch.tanh(self.a3(x))
        x = torch.tanh(self.b1(x))
        x = torch.tanh(self.b2(x))
        x = torch.tanh(self.b3(x))
        x = torch.tanh(self.c1(x))
        x = torch.tanh(self.c2(x))
        x = torch.tanh(self.d1(x))
        x = x.view(-1, 128)
        x = torch.tanh(self.last1(x))
        x = self.dropout(x)
        x = torch.tanh(self.last2(x))
        x = self.dropout(x)
        x = self.last3(x)
        return torch.tanh(x)


class regNet(nn.Module):
    def __init__(self):
        super(regNet, self).__init__()
        self.a = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3, 3), padding=1)
        self.b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.c = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.d = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2, 2))
        self.e = nn.Linear(128, 64)
        self.f = nn.Linear(64, 1)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.relu(self.a(x))
        x = torch.relu(self.b(x))
        x = torch.relu(self.c(x))
        x = torch.relu(self.d(x))
        x = x.view(-1, 128)
        x = torch.tanh(self.e(x))
        x = self.drop(x)
        x = torch.tanh(self.f(x))

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chess_dataset = TestDataset()
    train_loader = torch.utils.data.DataLoader(chess_dataset, batch_size=1024, shuffle=True)
    model = regNet()
    model.cuda()
    summary(model, (7, 8, 8))
    optimizer = optim.Adagrad(model.parameters(), lr=0.05, eps=1e-7)
    floss = nn.MSELoss()

    model.train()

    for epoch in range(20):
        all_loss = 0
        num_loss = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            data = data.float()

            target = target.float()
            optimizer.zero_grad()
            output = model(data)
            loss = floss(output, target)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            num_loss += 1
        print("\n%3d: %f" % (epoch, all_loss / num_loss))
        torch.save(model.state_dict(), "../nets/value.pth")

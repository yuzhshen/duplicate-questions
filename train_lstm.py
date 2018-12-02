import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import util as u
from torch.utils.data import Dataset, DataLoader
import constants as c
torch.manual_seed(1)


class MatrixDataset(Dataset):

    def __init__(self, save_batch):
        self.matrices = np.load('training/training_matrices_b{}.npy'.format(save_batch))
        self.labels = np.load('training/training_labels_b{}.npy'.format(save_batch))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'matrix':self.matrices[idx], 'label':self.labels[idx]}

class LstmNet(nn.Module):

    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size = c.WORD_EMBED_DIM,
                            hidden_size = 100,
                            num_layers = 2)
        self.fc1 = nn.Linear(100,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc3 = nn.Linear(25,2)
    
    def forward(self, x):
        x = self.lstm(x)
        x = x[-1]
        x = nn.LeakyReLU()(self.bn1(self.fc1(x)))
        x = nn.LeakyReLU()(self.bn2(self.fc2(x)))
        x = nn.Softmax(dim=1)(self.fc3(x))
        return x

if __name__ == '__main__':

    net = LstmNet().cuda()
    opt = optim.Adam(net.parameters())
    loss_func = nn.CrossEntropyLoss()

    net.train()
    for e in range(c.NUM_EPOCHS):
        print('Epoch {}'.format(e+1))
        for i in range(12):
            print('')
            dl = DataLoader(MatrixDataset(i), batch_size=c.TRAIN_BATCH_SIZE, shuffle=True)
            for batch in dl:
                data = batch['matrix'].cuda()
                label = batch['label'].cuda()
                print(label)
                print(type(label))
                raise Exception()
                output = net(data)
                # convert label to one-hot
                loss = loss_func(output, target)
                loss.backward()
                opt.step()
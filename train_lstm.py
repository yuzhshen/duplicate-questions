import torch, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import util as u
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import constants as c
torch.manual_seed(1)


class MatrixDataset(Dataset):

    def __init__(self, is_val):
        self.matrices = np.load('training/tr_matrices_good_b0.npy'.format(save_batch))      #TODO: change to best
        self.labels = np.load('training/tr_labels_good_b0.npy'.format(save_batch))          #TODO: change to best
        N = len(self.labels)
        split = int(0.9*N)
        if is_val:
            self.matrices = self.matrices[split:]
            self.labels = self.labels[split:]
        else:
            self.matrices = self.matrices[:split]
            self.labels = self.labels[:split]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'matrix':self.matrices[idx], 'label':self.labels[idx]}

class LstmNet(nn.Module):

    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = nn.LSTM(input_size = c.SENT_INCLUSION_MAX,
                            hidden_size = 100,
                            num_layers = 2)
        self.fc1 = nn.Linear(100,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50,25)
        self.bn2 = nn.BatchNorm1d(25)
        self.fc3 = nn.Linear(25,2)
    
    def forward(self, x):
        x, (_,_) = self.lstm(x)
        x = x[-1]
        x = nn.LeakyReLU()(self.bn1(self.fc1(x)))
        x = nn.LeakyReLU()(self.bn2(self.fc2(x)))
        x = nn.Softmax(dim=1)(self.fc3(x))
        return x

if __name__ == '__main__':

    net = LstmNet().cuda()
    opt = optim.Adam(net.parameters())
    loss_func = nn.CrossEntropyLoss()

    print('Loading datasets...')
    train_dl = DataLoader(MatrixDataset(is_val=False), batch_size=c.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(MatrixDataset(is_val=True), batch_size=c.BATCH_SIZE, shuffle=True)

    print('Training...')
    for e in range(c.NUM_EPOCHS):
        net.train()
        batch_num = 0
        for batch in train_dl:
            opt.zero_grad()
            data = np.swapaxes(batch['matrix'],0,1)
            data = torch.tensor(data, dtype=torch.float32).cuda()
            target = torch.tensor(batch['label'],dtype=torch.int64).cuda()
            preds = net(data)
            loss = loss_func(preds, target)
            loss.backward()
            opt.step()
            acc = (preds.max(1)[1]==target).sum().float()/len(preds)
            if batch_num%10==0:
                print('Epoch:{}\tBatch:{}\tLoss:{}\tAccuracy:{}'.format(e+1, batch_num, loss, acc))
            batch_num+=1
            
        print('Calculating validation statistics...')
        with torch.no_grad():
            net.eval()
            accs = []
            f1s = []
            for batch in val_dl:
                if len(batch['label'])==c.BATCH_SIZE:
                    data = np.swapaxes(batch['matrix'],0,1)
                    data = torch.tensor(data, dtype=torch.float32).cuda()
                    target = torch.tensor(batch['label'],dtype=torch.int64).cuda()
                    preds = net(data)
                    # print(data)
                    # print(target)
                    # print(preds.max(1)[1])
                    # print(target)
                    # raise NotImplementedError()
                    accs.append(float((preds.max(1)[1]==target).sum().float()/len(preds)))
                    f1s.append(f1_score(preds.max(1)[1], target))
            acc_score = sum(accs)/len(accs)
            f1_score = sum(f1s)/len(f1s)
            print('End of Epoch {}\tAccuracy:{}\tF1:{}'.format(e,acc_score,f1_score))

    torch.save(net,'net.pt')

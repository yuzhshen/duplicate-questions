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

    for e in range(c.NUM_EPOCHS):
        net.train()
        batch_num = 0
        for i in range(1): #TODO: CHANGE BACK TO 11 #TODO: CHANGE BACK TO 11 #TODO: CHANGE BACK TO 11 #TODO: CHANGE BACK TO 11 #TODO: CHANGE BACK TO 11
            print('Loading data fragment {}...'.format(i))
            dl = DataLoader(MatrixDataset(i), batch_size=c.TRAIN_BATCH_SIZE, shuffle=True)
            for batch in dl:
                data = np.swapaxes(batch['matrix'],0,1)
                data = torch.tensor(data, dtype=torch.float32).cuda()
                target = torch.tensor(batch['label'],dtype=torch.int64).cuda()
                print([max(x) for x in data])
                print(target)
                raise NotImplementedError()
                preds = net(data)
                loss = loss_func(preds, target)
                loss.backward()
                opt.step()
                acc = (preds.max(1)[1]==target).sum().float()/len(preds)
                if batch_num%10==0:
                    print('Epoch:{}\tBatch:{}\tLoss:{}\tAccuracy:{}'.format(e+1, batch_num, loss, acc))
                batch_num+=1
        print('Calculating validation statistics...')
        dl = DataLoader(MatrixDataset(11), batch_size=c.ELSE_BATCH_SIZE)
        with torch.no_grad():
            net.eval()
            accs = []
            f1s = []
            for batch in dl:
                if len(batch['label'])==c.ELSE_BATCH_SIZE:
                    data = np.swapaxes(batch['matrix'],0,1)
                    data = torch.tensor(data, dtype=torch.float32).cuda()
                    target = torch.tensor(batch['label'],dtype=torch.int64).cuda()
                    preds = net(data)
                    print(preds.max(1)[1])
                    print(target)
                    raise NotImplementedError()
                    accs.append(float((preds.max(1)[1]==target).sum().float()/len(preds)))
                    f1s.append(f1_score(preds.max(1)[1], target))
            acc_score = sum(accs)/len(accs)
            f1_score = sum(f1s)/len(f1s)
            print('End of Epoch {}\tAccuracy:{}\tF1:{}'.format(e,acc_score,f1_score))

    torch.save(net,'net.pt')

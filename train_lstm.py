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
        
        return {'matrix':np.swapaxes(self.matrices[idx],0,1), 'label':self.labels[idx]}

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
        print(x.size())
        x = x[-1]
        print(x.size())
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
                data = torch.tensor(batch['matrix'],dtype=torch.float32).cuda()
                label = torch.tensor(batch['label'],dtype=torch.int32).cuda()
                print(data.size(), data.dtype)
                print(label.size(), label.dtype)
                one_hot_list = [[1-x,0+x] for x in label]
                target = torch.tensor(one_hot_list).float()
                preds = net(data)
                # convert label to one-hot
                loss = loss_func(preds, target)
                loss.backward()
                opt.step()
                acc = (preds.max(1)[1]==target.max(1)[1]).sum().float()/len(preds)
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
                if len(batch)==c.ELSE_BATCH_SIZE:
                    preds = net(torch.tensor(batch['matrix'],dtype=torch.float32))
                    labels = torch.tensor(batch['label'],dtype=torch.int32)
                    accs.append(float((preds.max(1)[1]==batch['label']).sum().float()/len(preds)))
                    f1s.append(f1_score(preds.max(1)[1], batch['label']))
            f1_score = sum(f1s)/len(f1s)
            acc_score = sum(accs)/len(accs)
            print('End of Epoch {}\tAccuracy:{}\tF1:{}'.format(e,acc_score,f1_score))

    torch.save(net,'net.pt')

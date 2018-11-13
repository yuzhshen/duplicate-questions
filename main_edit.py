import torch, bcolz, pickle
import numpy as np
import pandas as pd
import util_edit as u
import constants as c
from torch import nn
from torch.utils.data import Dataset, DataLoader


#TODO: refactor code to separate files

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gru = nn.GRU(
            input_size = c.WORD_EMBED_DIM,
            hidden_size = 100,
            num_layers = 1,
            dropout = 0.2,
            bidirectional = False
        )
        self.linear1 = nn.Linear(in_features = 400, out_features = 100)
        self.linear2 = nn.Linear(in_features = 100, out_features = 2)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, q1, q2):
        q1_emb = self.gru(q1)[0][-1,:,:] # batch_size * hidden_size
        q2_emb = self.gru(q2)[0][-1,:,:]
        feature_concat = torch.cat([q1_emb, q2_emb, q1_emb-q2_emb, q1_emb*q2_emb], dim=1)
        hl1_out = nn.ReLU()(nn.BatchNorm1d(100)(self.linear1(feature_concat)))
        output = self.softmax(self.linear2(hl1_out))
        return output


class QuestionsDataset(Dataset):

    def __init__(self, data_frame, *, is_val, glove):
        split_point = len(data_frame)//c.NUM_FOLDS
        self.data = data_frame[:split_point] if is_val else data_frame[split_point:]
        self.glove = glove

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q1, q2, is_dup = self.data.loc[idx,['question1','question2','is_duplicate']]
        q1, q2 = self.sent_to_embed(q1), self.sent_to_embed(q2)
        return {'q1':q1, 'q2':q2, 'is_dup':self.one_hot(is_dup)}
    
    def one_hot(self, is_dup):
        if is_dup=='0':
            return torch.tensor([1,0])
        elif is_dup=='1':
            return torch.tensor([0,1])
        else:
            raise Exception('Unexpected label encountered: {}'.format(is_dup))

    def sent_to_embed(self, sentence):
        sent_list = sentence.split()
        for e, word in enumerate(sent_list):
            if word not in self.glove:
                sent_list[e] = '<UNK>'
        embed = torch.zeros([c.SENT_INCLUSION_MAX, c.WORD_EMBED_DIM])
        for idx in range(len(sent_list)):
            if sent_list[idx] == '<UNK>':
                embed[idx] = torch.zeros(c.WORD_EMBED_DIM)
            else:   
                embed[idx] = self.glove[word]
        return embed



# TODO: ensure cleaned data (lemmatized/preprocessed)
print('Building datasets...')
# load data objects into dataframe (should be clean at this point)
data_frame = pd.read_csv(c.TRAIN_VAL_PATH)
data_frame = u.exclude_sents(data_frame)
data_frame = data_frame.reset_index(drop=True)

# build glove dict, save files to disk if they don't already exist
print('Loading GloVe...')
u.embeddings_to_disk()
vectors_path = c.GLOVE_FILEPATH+'.vectors.dat'
words_path = c.GLOVE_FILEPATH+'.words.pkl'
idx_path = c.GLOVE_FILEPATH+'.idx.pkl'
vectors = bcolz.open(vectors_path)[:]
words = pickle.load(open(words_path, 'rb'))
word2idx = pickle.load(open(idx_path, 'rb'))
print(vectors.shape, len(words), len(word2idx))
glove = {w: torch.tensor(vectors[word2idx[w]]) for w in words}

# Create train/val datasets
train_dataset = QuestionsDataset(data_frame, is_val=False, glove=glove)
val_dataset = QuestionsDataset(data_frame, is_val=True, glove=glove)

# build embedding dict (values should be torch tensors)
# embed_dict = {}
# for elt in train_dataset:
#     words = elt['q1'].split()+elt['q2'].split()
#     for word in words:
#         if word 

# augment both train and val data (flips, same)
# split and pad


# # data augmentation
# train_data, val_data = augmented(TRAIN_VAL_PATH, method='AUG_POOLED',fold_num=0)
# train, val = tfd.DataFrameDataset(train_data,fields), tfd.DataFrameDataset(val_data,fields)

# print('Building vocab...')
# TEXT.build_vocab(train, vectors='glove.6B.'+str(WORD_EMBED_DIM)+'d')  
# print('Done.')

# train_iter, val_iter = data.Iterator.splits(
#     (train, val),
#     batch_sizes=(TRAIN_BATCH_SIZE, ELSE_BATCH_SIZE),
#     repeat=False,
#     shuffle=True,
#     sort_key=lambda x: len(x.q1),
#     device=-1)

# print(type(train))

# vocab = TEXT.vocab
# embed = nn.Embedding(len(vocab), WORD_EMBED_DIM)
# embed.weight.data.copy_(vocab.vectors)

# # training process
# net = Model()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.0001)
# for epoch in range(NUM_EPOCHS):
#     for batch in train_iter:
#         q1, q2, labels = embed(getattr(batch,'q1')), embed(getattr(batch,'q2')), getattr(batch,'y')
#         optimizer.zero_grad()
#         preds = net(q1, q2)
#         labels = labels.view(-1)
#         loss = criterion(preds, labels)
#         # transform network output to labeled guess

#         acc_correct = int((preds.max(1)[1].view(-1)==labels).sum())
#         acc_total = TRAIN_BATCH_SIZE
#         print('Accuracy: '+str(acc_correct/acc_total))

#         loss.backward()
#         optimizer.step()
#     val_loss = 0
#     for batch in helper.HelperIterator(iterator = val_iter, fields = fields):
#         batch_q1, batch_q2 = embed(batch.q1), embed(batch.q2)
#         batch_y = net(batch_q1, batch_q2)
#         batch_target = batch.y.view(-1)
#         batch_size = batch_target.size()[0]
#         loss = criterion(batch_y, batch_target)
#         val_loss+=loss
#     print(val_loss)

        
        
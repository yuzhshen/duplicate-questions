import torch, pickle
import numpy as np
import pandas as pd
import util as u
from torch import nn
from torch.utils.data import Dataset, DataLoader


# get sentence vectors, if pad or unknown, assign 0 vector
# dataset composed of sent_len*sent_len matrix, with attached label yes/no 

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
import torch
from torch import nn
from torchtext import data
import spacy
import torchtext_helper as helper

# files
TRAIN_PATH = 'data/med_train.csv'
VAL_PATH = 'data/small_train.csv'
TEST_PATH = 'data/test.csv'
# representation
WORD_EMBED_DIM = 50
# training
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 32
ELSE_BATCH_SIZE = 128

# #TODO: refactor to separate files

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gru = nn.GRU(
            input_size = WORD_EMBED_DIM,
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

nlp = spacy.load('en')

def tokenizer(text):
    return [x.lemma_ for x in nlp(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, fix_length=50)
NUMBER = data.Field(sequential=False, use_vocab=False, preprocessing=int)
fields = [('pairID',NUMBER), ('id1',NUMBER), ('id2',NUMBER), ('q1',TEXT), ('q2',TEXT), ('y',NUMBER)]

train, val = data.TabularDataset.splits(
    path='', train=TRAIN_PATH, validation=VAL_PATH,
    format='csv', 
    skip_header=True,
    fields=fields
)
print('Building vocab...')
TEXT.build_vocab(train, vectors='glove.6B.'+str(WORD_EMBED_DIM)+'d')  
print('Done.')

train_iter, val_iter = data.Iterator.splits(
    (train, val),
    batch_sizes=(TRAIN_BATCH_SIZE, ELSE_BATCH_SIZE),
    repeat=False,
    sort_key=lambda x: len(x.q1),
    device=-1)

vocab = TEXT.vocab
embed = nn.Embedding(len(vocab), WORD_EMBED_DIM)
embed.weight.data.copy_(vocab.vectors)


# training process
net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), weight_decay=0.0001)
for epoch in range(NUM_EPOCHS):
    for batch in helper.HelperIterator(iterator = train_iter, fields = fields):
        optimizer.zero_grad()
        batch_q1, batch_q2 = embed(batch.q1), embed(batch.q2)
        batch_y = net(batch_q1, batch_q2)
        batch_target = batch.y.view(-1)
        loss = criterion(batch_y, batch_target)
        loss.backward()
        optimizer.step()
    val_loss = 0
    for batch in helper.HelperIterator(iterator = val_iter, fields = fields):
        batch_q1, batch_q2 = embed(batch.q1), embed(batch.q2)
        batch_y = net(batch_q1, batch_q2)
        batch_target = batch.y.view(-1)
        batch_size = batch_target.size()[0]
        loss = criterion(batch_y, batch_target)
        val_loss+=loss
    print(val_loss)

        
        
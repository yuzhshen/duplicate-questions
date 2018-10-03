from torch import nn
from torchtext import data
import spacy
import torchtext_helper as helper

# files
TRAIN_PATH = 'data/small_train.csv'
VAL_PATH = 'data/trivial_train.csv'
TEST_PATH = 'data/test.csv'
# representation
WORD_EMBED_DIM = 50
# training
NUM_EPOCHS = 10
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
    
    def forward(self, q1, q2):
        q1_emb = self.gru(q1)
        q2_emb = self.gru(q2)
        # TODO: similarity

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
TEXT.build_vocab(train, vectors='glove.6B.'+str(WORD_EMBED_DIM)+'d')  
 
train_iter, val_iter = data.Iterator.splits(
    (train, val),
    batch_sizes=(TRAIN_BATCH_SIZE, ELSE_BATCH_SIZE),
    repeat=False,
    device=-1)

vocab = TEXT.vocab
embed = nn.Embedding(len(vocab), WORD_EMBED_DIM)
embed.weight.data.copy_(vocab.vectors)

for epoch in range(NUM_EPOCHS):
    for batch in helper.HelperIterator(train_iter, fields):
        print(batch.q1.size())
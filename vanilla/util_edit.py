import re, io, bcolz, pickle, os.path
import numpy as np
import pandas as pd
import constants as c
from nltk.stem.wordnet import WordNetLemmatizer



def exclude_sents(data):
    """Returns: dataframe without pairs that include high-length and low-length questions

    - data: dataframe of question pairs
    """
    # splitting sentence strings
    data['question1'] = data['question1'].str.split()
    data['question2'] = data['question2'].str.split()
    # removing floats (NaN?)
    data = data.drop(data[data['question1'].apply(type)==float].index)  
    data = data.drop(data[data['question2'].apply(type)==float].index)
    # removing sentences that are too short/long
    q1_longs = data[data['question1'].apply(len)>c.SENT_INCLUSION_MAX].index
    q1_shorts = data[data['question1'].apply(len)<c.SENT_INCLUSION_MIN].index
    q2_longs = data[data['question2'].apply(len)>c.SENT_INCLUSION_MAX].index
    q2_shorts = data[data['question2'].apply(len)<c.SENT_INCLUSION_MIN].index
    index_list = q1_longs.union(q1_shorts).union(q2_longs).union(q2_shorts)
    data = data.drop(index_list)
    data['question1'] = data['question1'].apply(' '.join)
    data['question2'] = data['question2'].apply(' '.join)
    return data

def embeddings_to_disk():
    """Saves embedding data to disk.
    """

    vectors_path = c.GLOVE_FILEPATH+'.vectors.dat'
    words_path = c.GLOVE_FILEPATH+'.words.pkl'
    idx_path = c.GLOVE_FILEPATH+'.idx.pkl'

    if all([os.path.exists(vectors_path),os.path.exists(words_path),os.path.exists(idx_path)]):
        return

    print('Building GloVe on disk...')
    words = []
    idx = 0
    word2idx = {}

    vectors = bcolz.carray(np.zeros(1), rootdir=vectors_path, mode='w')
    with open(c.GLOVE_FILEPATH, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    
    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=vectors_path, mode='w')
    vectors.flush()
    pickle.dump(words, open(words_path, 'wb'))
    pickle.dump(word2idx, open(idx_path, 'wb'))
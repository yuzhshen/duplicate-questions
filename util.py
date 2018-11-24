import re, io, pickle, os.path
import numpy as np
import pandas as pd
import constants as c

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

    dict_path = c.GLOVE_FILEPATH+'.pydict.pkl'

    if os.path.exists(dict_path):
        return

    print('Saving GloVe dict to disk...')
    word2vect = {}
    with open(c.GLOVE_FILEPATH, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            word2vect[word] = vect
    
    pickle.dump(word2vect, open(dict_path, 'wb'))

def train_val_split(data,*,fold_num):
    """Shuffles and splits the data into training and validation sets.
        (fold_num is current iteration, num_folds is total number of folds)
    """
    shuffled_data = data.sample(n=len(data),random_state=27)
    folds_list = np.array_split(shuffled_data,c.NUM_FOLDS)
    val_data = folds_list[fold_num]
    del folds_list[fold_num]
    train_data = pd.concat(folds_list)
    return train_data, val_data

def augment_single_data(data):
    """- Returns: augmented dataframe
    """
    # data augmentation - Q1/Q2 swap
    swap = data.copy()
    swap['question1'],swap['question2']=swap['question2'].copy(),swap['question1'].copy()
    data = data.append(swap)
    # data augmentation - same question is duplicate of itself
    selfdup = pd.DataFrame(columns=data.columns)
    unique = data['question1'].unique() # only Q1 because already did Q1/Q2 swap augmentation
    selfdup['question1'], selfdup['question2'] = unique, unique
    selfdup['is_duplicate'] = [1]*len(unique)
    data = data.append(selfdup)
    # re-number indices
    data.index = range(len(data))
    # drop duplicate pairs
    data = data.drop_duplicates(subset=['question1','question2'])
    data = data.reset_index(drop=True)
    return data

def augmented(filepath):
    """Augments and excludes training data and excludes validation data
        - Returns: training and validation dataframes
    """
    data = pd.read_csv(filepath)
    data = data.drop(columns=['id','qid1','qid2','Unnamed: 0'])
    print('Excluding sentences by length...')
    data = exclude_sents(data)
    data = data.reset_index(drop=True)
    print('Creating training and validation sets...')
    train_data, val_data = train_val_split(data,fold_num=0)
    print("Augmenting training data...")
    train_data = augment_single_data(train_data)
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    return train_data, val_data

def vector_list(glove_dict, sent_str):
    """Turns sentence string into a list of word embeddings. Padding and unknowns are zero vectors.
        - Returns: list of numpy arrays
    """
    res = []
    for word in sent_str.split(' '):
        if word in glove_dict:
            res.append(glove_dict[word])
        else:
            res.append(np.zeros((c.WORD_EMBED_DIM)))
    if len(res) > c.SENT_INCLUSION_MAX:
        raise Exception('Unexpected number of vectors in vector_list')
    elif len(res) < c.SENT_INCLUSION_MAX:
        num_needed = c.SENT_INCLUSION_MAX - len(res)
        more_list = [np.zeros((c.WORD_EMBED_DIM))]*num_needed
        res = more_list+res
    return res

def custom_entropy(x):
    x = x/x.sum()
    return -np.sum(x*np.nan_to_num(np.log(x)))
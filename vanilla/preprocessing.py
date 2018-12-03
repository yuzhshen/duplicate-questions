import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer

NUM_FOLDS = 10
SENT_INCLUSION_MIN = 3
SENT_INCLUSION_MAX = 50
WNL = 

def lemmatizer(word):
    """Returns: lemmatized word if word >= length 5
    """
    wnl = WordNetLemmatizer()
    if len(word)<4:
        return word
    return wnl.lemmatize(wnl.lemmatize(word, "n"), "v")

def clean_string(string): # From kaggle-quora-dup submission
    """Returns: cleaned string, with common token replacements and lemmatization
    """
    string = string.lower().replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will").replace("=", " equal ").replace("+", " plus ")
    string = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', string)
    string = re.sub(r"([0-9]+)000000", r"\1m", string)
    string = re.sub(r"([0-9]+)000", r"\1k", string)
    string = ' '.join([lemmatizer(w) for w in string.split()])
    return string

def save_clean_data():
    """Saves clean data to file.
    """
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    print ('Preprocessing train Q1s...')
    train["question1"] = train["question1"].fillna("").apply(clean_string)
    print ('Preprocessing train Q2s...')
    train["question2"] = train["question2"].fillna("").apply(clean_string)
    print ('Preprocessing test Q1s...')
    test["question1"] = test["question1"].fillna("").apply(clean_string)
    print ('Preprocessing test Q2s...')
    test["question2"] = test["question2"].fillna("").apply(clean_string)
    print ('Storing data in CSV format...')
    train.to_csv('../data/train_clean.csv')
    test.to_csv('../data/test_clean.csv')
    print ('Data cleaned.')

def visualize_data(csvfilepath):
    data = pd.read_csv(csvfilepath)
    all_questions = data['question1'].append(data['question2'])
    all_split = all_questions.fillna("").apply(lambda x: x.split() if isinstance(x,str) else print(x))
    all_lens = all_split.str.len()
    plt.hist(all_lens,bins=200)
    plt.xlabel('Length (in words)')
    plt.ylabel('Frequency')
    plt.show()

def get_example_sents(csvfilepath,length):
    data = pd.read_csv(csvfilepath)
    all_questions = data['question1'].append(data['question2'])
    all_split = all_questions.fillna("").apply(lambda x: x.split() if isinstance(x,str) else print(x))
    all_lens = all_split.str.len()
    all_questions = all_questions[all_lens==length]
    chosen = np.random.choice(all_questions.index.values, 5)
    chosen_sents = all_questions.ix[chosen]
    for _, val in chosen_sents.iteritems():
        print (val)
        print()

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
    q1_longs = data[data['question1'].apply(len)>SENT_INCLUSION_MAX].index
    q1_shorts = data[data['question1'].apply(len)<SENT_INCLUSION_MIN].index
    q2_longs = data[data['question2'].apply(len)>SENT_INCLUSION_MAX].index
    q2_shorts = data[data['question2'].apply(len)<SENT_INCLUSION_MIN].index
    index_list = q1_longs.union(q1_shorts).union(q2_longs).union(q2_shorts)
    data = data.drop(index_list)
    data['question1'] = data['question1'].apply(' '.join)
    data['question2'] = data['question2'].apply(' '.join)
    return data

def train_val_split(data,fold_num):
    """Shuffles and splits the data into training and validation sets.
        (fold_num is current iteration, num_folds is total number of folds)
    """
    shuffled_data = data.sample(n=len(data),random_state=27)
    folds_list = np.array_split(shuffled_data,NUM_FOLDS)
    val_data = folds_list[fold_num]
    del folds_list[fold_num]
    train_data = pd.concat(folds_list)
    return train_data, val_data

def augment_single_data(data):
    """- data: pd.DataFrame
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
    return data

def augmented(filepath,*,method,fold_num):
    """Methods:
        - AUG_POOLED: augment, then train_val_split
        - AUG_SEPARATE: train_val_split, then augment each
        - AUG_TRAIN: only augment training data
    """
    print("Augmenting data...")
    data = pd.read_csv(filepath)
    data = exclude_sents(data)
    if method=='AUG_SEPARATE':
        train_data, val_data = train_val_split(data,fold_num)
        train_data = augment_single_data(train_data)
        val_data = augment_single_data(val_data)
        return train_data, val_data
    if method=='AUG_POOLED':
        data = augment_single_data(data)
        train_data, val_data = train_val_split(data,fold_num)
        return train_data, val_data
    if method=='AUG_TRAIN':
        train_data, val_data = train_val_split(data,fold_num)
        train_data = augment_single_data(train_data)
        return train_data, val_data

if __name__=="__main__":
    train,val = augmented('../data/train_clean.csv',method='AUG_POOLED')
    print ('train')
    print(train.groupby('is_duplicate').count())
    print('val')
    print(val.groupby('is_duplicate').count())
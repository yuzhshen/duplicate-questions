import torch, pickle, scipy.stats, scipy.optimize
import numpy as np
import pandas as pd
import util as u
import constants as c
from torch import nn
from torch.utils.data import Dataset, DataLoader

# for each training example:
#   get sentence vectors, if pad or unknown, assign 0 vector
#   do linear programming
#   save dataset composed of sent_len*sent_len matrix and attached label
# use entropy 

# TODO: ensure cleaned data (lemmatized/preprocessed)
print('Building dataframe...')
# load data objects into dataframe (should be clean at this point)
train_df, val_df = u.augmented(c.TRAIN_VAL_PATH)

# build glove dict, save files to disk if they don't already exist
print('Loading GloVe...')
u.embeddings_to_disk()
word2vect = pickle.load(open(c.GLOVE_FILEPATH+'.pydict.pkl', 'rb'))

print('Solving constrained optimizations...')
matrix_volume = []
label_volume = []
save_batch = 0
ex_per_batch = 10000
total_iters = len(train_df)
for idx in range(100000):
    if idx%10000==0:
        print('Iteration {} / {}'.format(idx+1, total_iters))
    goal_str = train_df.loc[idx,:]['question1']
    use_str = train_df.loc[idx,:]['question2']
    goal_vecs = u.vector_list(word2vect, goal_str)
    use_vecs = np.array(u.vector_list(word2vect, use_str))
    matrix = np.zeros((c.SENT_INCLUSION_MAX,c.SENT_INCLUSION_MAX))

    for g_idx in range(len(goal_vecs)):
        goal_vec = goal_vecs[g_idx]
        if (goal_vec == np.zeros((c.WORD_EMBED_DIM))).all():
            matrix[g_idx] = np.zeros((c.SENT_INCLUSION_MAX))
        else:
            objective = lambda weights: u.custom_entropy(np.array(weights))
            init_guess = [0.0001]*c.SENT_INCLUSION_MAX
            cons_func1 = lambda weights: np.array(weights).dot(use_vecs)-goal_vec+0.001
            cons_func2 = lambda weights: -np.array(weights).dot(use_vecs)+goal_vec+0.001
            constraint1 = {'type':'ineq','fun':cons_func1}
            constraint2 = {'type':'ineq','fun':cons_func2}
            bound = scipy.optimize.Bounds(0.,1.)
            res = scipy.optimize.minimize(objective, init_guess, 
                                        method='SLSQP', 
                                        constraints=[constraint1,constraint2],
                                        bounds=bound)
            matrix[g_idx] = np.nan_to_num(res.x)
    matrix_volume.append(matrix)
    label_volume.append(train_df.loc[idx,:]['is_duplicate'])
    if (idx+1)//ex_per_batch > idx//ex_per_batch or idx==total_iters-1:
        print('Writing batch {} data to disk...'.format(save_batch))
        matrix_volume = np.array(matrix_volume)
        label_volume = np.array(label_volume)
        np.save('training/tr_matrices_best_b{}.npy'.format(save_batch), matrix_volume)
        np.save('training/tr_labels_best_b{}.npy'.format(save_batch), label_volume)
        matrix_volume = []
        label_volume = []
        save_batch+=1

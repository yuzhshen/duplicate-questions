# representaton hyperparameters
WORD_EMBED_DIM = 50

# training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_FOLDS = 10 # determines size of validation set

# preprocessing
SENT_INCLUSION_MIN = 3
SENT_INCLUSION_MAX = 30

# filepaths
TRAIN_VAL_PATH = 'data/train_clean_subset.csv' # TODO: change back to full set
TEST_PATH = 'data/test.csv'
GLOVE_FILEPATH = 'data/embeddings/glove.6B.'+str(WORD_EMBED_DIM)+'d.txt'
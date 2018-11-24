### iFAQ
Duplicate question detection for document question answering

- Run create_train.py to save a dataset



- Notes about linear program
    - if padding or unknown, use 0 vector
    - for the optimization objective, use entropy, but most improvement would probably be found here.
        - entropy is good metric assuming that scaled word vectors have related meanings, but as the GloVe vector space is rather abstract I don't think that this is the case. I will have to look further into it, maybe utilizing entropy and trying to make weighted sum close to 1 would be a better technique.
    - reduced max sent_len to 30 to conserve resources, but those higher word count sentences are infrequent anyway, and unaugmented training question count only decreases from 362762 to 355473.
    - couldn't use exact equality, had to bound
    - constraint bounds explanation
    - explain handling of when goal is zero - no information so don't weight
    - future expansion, incorporate an additional input that accounts for prior knowledge of document frequency of word so that common words matching is trivialized
    - zero initialization screws up the optimization method
import spacy
import torch

nlp = spacy.load('en')

def tokenizer(text):
    return [x.lemma_ for x in nlp(text)]
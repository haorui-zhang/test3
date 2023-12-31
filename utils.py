import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from nltk.corpus import wordnet

import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

    
def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    
    ## Reference Chatgpt and Penn Treebank Method
    

    postag = pos_tag(word_tokenize(example['text']))
    
    trans_result = []
    for w, t in postag:
        if t.startswith('J'):
            pos_idx = wordnet.ADJ
        elif t.startswith('V'):
            pos_idx = wordnet.VERB
        elif t.startswith('N'):
            pos_idx = wordnet.NOUN
        elif t.startswith('R'):
            pos_idx = wordnet.ADV
        else:
            pos_idx = ''

        if pos_idx:
            syn = set()
            for i in wordnet.synsets(w, pos=pos_idx):
                for lem in i.lemmas():
                    if lem.name() != w:
                        syn.add(lem.name().replace('_', ' '))
            result = random.choice(list(syn)) if syn else w
        else:
            result = w
        trans_result.append(result)

    example['text'] = ' '.join(trans_result)
    return example
##### YOUR CODE ENDS HERE ######


# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Filename: Data_Load.py 
# Developed: Hancheol Moon
# Purpose: Load data
# ----------------------------------------------------------------
# ----------------------------------------------------------------

import sys, os
import numpy as np
import json
import pickle
import pdb

import re
import argparse
import datetime
import random
import copy

from torch.utils import data
import torch
import torch.nn as nn

#def Tensor_Indexing(torch_tensor, idx_list):
#    return torch_tensor[idx_list]
def ScoreMask(scores, batch_docs_len, device):
    score_mask = torch.ones_like(scores).to(device).requires_grad_(False)
    for i, valid_length in enumerate(batch_docs_len):
        score_mask[i, valid_length:, :] = 0
    return score_mask

def Unpairing_Pos_Neg_new(doc_batch):

    pos_batch = []
    neg_batch = []
    pos_eos = []
    neg_eos = []
    for pos_neg in doc_batch:
        pos_eos_position = [pos for pos, eos in enumerate(pos_neg[0]) if eos=="</s>"]
        pos = [(pos_neg[0])[:pos_eos_position[i]+1] if i==0 else (pos_neg[0])[pos_eos_position[i-1]+1:pos_eos_position[i]+1] for i in range(len(pos_eos_position)) ]
        pos_batch.append(pos) #pos is a list of sentences containing lists of tokens of sentences
        pos_eos.append(pos_eos_position)

        neg_eos_position = [neg for neg, eos in enumerate(pos_neg[1]) if eos=="</s>"]
        neg = [(pos_neg[1])[:neg_eos_position[i]+1] if i==0 else (pos_neg[1])[neg_eos_position[i-1]+1:neg_eos_position[i]+1] for i in range(len(neg_eos_position)) ]
        neg_batch.append(neg)
        neg_eos.append(neg_eos_position)
    return pos_batch, neg_batch, pos_eos, neg_eos

def Load_File(loadpath, types):
    if types=='pickle':
        with open(loadpath, 'rb') as fp:
            data =pickle.load(fp)
        return data
    
    elif types=='json':
        with open(loadpath, 'r') as fout:
            data = json.load(fout) 
        return data 
    elif types=='npy':
        data = np.load(loadpath)
        return data
    else:
        print("Type Error")
        

def Batch2Idx_new(batch, word2idx):
    """
    - Input should be a list of list of with tokens 
        -->[doc->sentence->tokens]
    """
    batch_list = []
    for doc in batch:
        doc_list = []
        for sent in doc:
            sent_list = []
            for word in sent:
                try:
                    idx = word2idx[word]
                    sent_list.append(idx)
                except:
                    idx = word2idx['<unk>']
                    sent_list.append(idx)
            doc_list.append(sent_list)
        batch_list.append(doc_list)
    return batch_list

def Batch_Padding(batch, batch_first, pad_token):
    """
    Sentences that are shorter than the max sentence length will be padded
    """
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch], batch_first=True, padding_value=pad_token)
    return padded_batch

def pad_sentences(docs, docslen, padding):
    '''
    makes all the doc same length (#sentences) with padding
    '''
    maxlen = max([max(docslen[i]) for i in range(len(docslen))])
    #print(maxlen)
    def func(i, j):
        # doc i, sentence j
        needed = maxlen - len((docs[i])[j])
        return (docs[i])[j]+[padding for k in range(needed)]
    
    return [[func(i, j) for j in range(len(docs[i]))] for i in range(len(docs))]

def score_masking(scores, output_masking, args):
    score_squeezed = scores.squeeze()
    score_mask = torch.tensor(output_masking, dtype=torch.float, device=args.device)
    return score_squeezed*score_mask

def batch_sentences_length(batch_idx):
    batch_sentences_len = []
    for i in range(len(batch_idx)):
        sentences_len = [len((batch_idx[i])[j]) for j in range(len(batch_idx[i]))]
        batch_sentences_len.append(sentences_len)
    return batch_sentences_len

def pad_senteces_len(X, maxlen):
    new_X = np.ones((len(X), maxlen), dtype=int)
    for i in range(len(X)):
        if (len(X[i]) > maxlen):
            new_X[i] = X[i][0:maxlen]
        else:
            new_X[i][0:len(X[i])] = X[i]
    return new_X


def Batch_Preprocessing_new(batch, word2idx, batchs_size, batch_first=True, pad_token="<pad>"):
    """
    1) Batch is the raw words (not indice)
    2) Batch should be at the first dim
    3) Batch should be sorted in the descending order to use the "packed padded sequence function"

    - batch_size: size of mini-batch
    """
    pad_token = word2idx[pad_token]
    batch_idx = Batch2Idx_new(batch, word2idx)
    batch_docs_len = [len(batch_idx[i]) for i in range(len(batch_idx))] #num of sentences in each docs 1D list [Doc]
    batch_sentences_len = batch_sentences_length(batch_idx) #num of tokens in each sentences in each docs [doc->#tokens_in_sent]
    batch_idx_padded = pad_sentences(batch_idx, batch_sentences_len, pad_token) #makinbg every sent same lengths
    padded_batch = Batch_Padding(batch_idx_padded, batch_first=True, pad_token=pad_token) #making every doc same length. 3D Tensor [doc->sentences->tokens]
    modified_batch_sentences_len = pad_senteces_len(batch_sentences_len, padded_batch.size(1)) 

    return padded_batch, batch_docs_len, batch_sentences_len, modified_batch_sentences_len

class Batch_Generator_Insertion():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, dirname, args, test=False):
        """
        - dirname: file directory name
        - file_types: json or pickle...
        - batch_size: mini batch size
        - shuffle: shuffle the dataset?
        - batch_length = length of each document in a mini-batch
        """
        self.dirname = dirname
        self.file_types = args.file_types
        self.test = test

        if self.test==True:
            self.batch_size = args.batch_size_test
        else:
            self.batch_size = args.batch_size

        self.shuffle = args.shuffle
        self.n_window = args.n_window

    def __iter__(self):
        items = os.listdir(self.dirname)

        insertion_item = []
        for item in items:
            dataname = item[:8]
            if dataname not in insertion_item:
                insertion_item.append(dataname)

        insertion_item = [data+".pos.text_"+str(self.n_window)+"_1" for data in insertion_item]

        if self.shuffle == True:
            print("Shuffle the dataset")
            #random.seed(0) # fix the randomness
            random.shuffle(items)
        else:
            print("Without shuffle")

        batch = []
        batch_fname = []
        batch_length = []
        for fname in insertion_item: 
            loadpath = os.path.join(self.dirname, fname)
            batch_file = Load_File(loadpath, self.file_types)
            batch.append(batch_file)
            batch_fname.append(fname)
            batch_length.append(len(batch_file[0]))

            if len(batch)==self.batch_size:
                yield batch, batch_length, batch_fname
                batch = [] # make it batch empty for the next iteration
                batch_fname = [] 
                batch_length = [] 

class Batch_Generator():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, dirname, args, test=False):
        """
        - dirname: file directory name
        - file_types: json or pickle...
        - batch_size: mini batch size
        - shuffle: shuffle the dataset?
        - batch_length = length of each document in a mini-batch
        """
        self.dirname = dirname
        self.file_types = args.file_types
        self.test = test

        if self.test==True:
            self.batch_size = args.batch_size_test
        else:
            self.batch_size = args.batch_size

        self.shuffle = args.shuffle

    def __iter__(self):
        items = os.listdir(self.dirname)

        if self.shuffle == True:
            print("Shuffle the dataset")
            #random.seed(0) # fix the randomness
            random.shuffle(items)
        else:
            print("Without shuffle")

        batch = []
        batch_fname = []
        batch_length = []
        for fname in items: 
            loadpath = os.path.join(self.dirname, fname)
            batch_file = Load_File(loadpath, self.file_types)
            batch.append(batch_file)
            batch_fname.append(fname)
            batch_length.append(len(batch_file[0]))

            if len(batch)==self.batch_size:
                yield batch, batch_length, batch_fname
                batch = [] # make it batch empty for the next iteration
                batch_fname = [] 
                batch_length = [] 


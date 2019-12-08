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
import datetime
#from slackclient import SlackClient

def Order_Creator(pos_batch, neg_batch, batch_docs_len, device):
    label = []
    max_len = max(batch_docs_len)
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos!=neg:
                start = max(i-2, 0)
                end = min(i+3, len(pos_doc))
                for j in range(start,end):
                    if pos_doc[j]==neg:
                        #print(j)
                        label_temp.append(j)
                        break
            else:
                label_temp.append(i)

        if len(label_temp)<max_len:
            pad_len = max_len - len(label_temp)
            for pad in range(pad_len):
                label_temp.append(i+pad+1)
                
        label.append(torch.LongTensor(label_temp).to(device))
            
    return label


def Order_Creator_old(pos_batch, neg_batch, batch_docs_len, device):
    label = []
    max_len = max(batch_docs_len)
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos!=neg:
                for j, pos_j in enumerate(pos_doc):
                    if pos_j==neg:
                        label_temp.append(j)
                        break

            else:
                label_temp.append(i)

        if len(label_temp)<max_len:
            pad_len = max_len - len(label_temp)
            for pad in range(pad_len):
                label_temp.append(i+pad+1)
                
        label.append(torch.LongTensor(label_temp).to(device))
            
    return label

def Label_Creator(pos_batch, neg_batch, device):
    label = []
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos!=neg:
                for j, pos_j in enumerate(pos_doc):
                    if pos_j==neg:
                        label_temp.append(j)

            else:
                label_temp.append(i)

        label.append(torch.LongTensor(label_temp).to(device))
            
    return label


def Label_Creator_new(pos_batch, neg_batch, device):
    label = []
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos!=neg:
                start = max(i-2, 0)
                end = min(i+3, len(pos_doc))
                for j in range(start,end):
                    if pos_doc[j]==neg:
                        #print(j)
                        label_temp.append(j)
                        break

            else:
                label_temp.append(i)

        label.append(torch.LongTensor(label_temp).to(device))
            
    return label

def Label_Creator_Global(pos_batch, neg_batch, device):
    label = []
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = []
        tracker = np.zeros(len(pos_doc))
        #print(tracker)
        for i, (pos, neg) in enumerate(zip(pos_doc, neg_doc)):
            if pos!=neg:
                for j in range(len(pos_doc)):
                    if pos_doc[j]==neg and tracker[j]==0:
                        #print(j)
                        label_temp.append(j)
                        tracker[j] = 1 
                        break

            else:
                label_temp.append(i)
                tracker[i] = 1

        label.append(torch.LongTensor(label_temp).to(device))
        #print(tracker)
    return label

def Label_Creator_Inverse(pos_batch, neg_batch, device):
    label = []
    for pos_doc, neg_doc in zip(pos_batch, neg_batch):
        label_temp = [i for i in range(len(pos_doc))][::-1]
        label.append(torch.LongTensor(label_temp).to(device))
        
    return label

#    pos_label = torch.ones(label.size(), dtype=torch.float).to(device)
#    neg_label = label.to(device)
#
#    return pos_label, neg_label

class SentenceBatch2():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, docu_batch, n_sent):
        self.docu_batch = docu_batch
        self.n_sent = n_sent

    def __iter__(self):

        batch = []
        for sent_batch in self.docu_batch: 
            batch.append(sent_batch)

            if len(batch)==self.n_sent:
                yield batch
                batch = [] # make it batch empty for the next iteration
        yield batch

class SentenceBatch():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, docu_batch):
        self.docu_batch = docu_batch

    def __iter__(self):

        batch = []
        for sent_batch in zip(*self.docu_batch): 
            batch.append(sent_batch)

            yield batch
            batch = [] # make it batch empty for the next iteration
        yield batch

def slack_message(message):
    channel = 'DHEC22JPM'
    #channel = 'CHVQ42M98'
    
    token = 'xoxp-607448253271-592410068819-621287707894-87eef95d4097f4026f3c5447449d3bcf'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel, 
                text=message, username='Python Message',
                icon_emoji=':robot_face:')

def Print_Args(args):
    """
    Print all arguments in argparser
    """
    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    now = datetime.datetime.now()
    print(f"||Experiment Date:{now.year}-{now.month}-{now.day}||")
    print("Arguments List: \n")
    for arg in vars(args): 
        print(f"- {arg}: {getattr(args, arg)}")

    print("---------------------------------------------------------")
    print("---------------------------------------------------------\n")

#def Tensor_Indexing(torch_tensor, idx_list):
#    return torch_tensor[idx_list]
def ScoreMask(scores, batch_docs_len, device):
    score_mask = torch.ones_like(scores).to(device).requires_grad_(False)
    for i, valid_length in enumerate(batch_docs_len):
        score_mask[i, valid_length:, :] = 0
    return score_mask

def Unpairing_Pos_Neg(doc_batch):
    pos_batch = []
    neg_batch = []
    for pos_neg in doc_batch:
        pos_batch.append(pos_neg[0])
        neg_batch.append(pos_neg[1])

    return pos_batch, neg_batch

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

#def Batch_Preprocessing_ELMo(batch, word2idx, batchs_size, batch_first=True, pad_token="<pad>"):
#    """
#    1) Batch is the raw words (not indice)
#    2) Batch should be at the first dim
#    3) Batch should be sorted in the descending order to use the "packed padded sequence function"
#
#    - batch_size: size of mini-batch
#    """
#    pad_token = word2idx[pad_token]
##    batch_idx = Batch2Idx_new(batch, word2idx)
#    batch_docs_len = [len(batch[i]) for i in range(len(batch))] #num of sentences in each docs 1D list [Doc]
#    batch_sentences_len = batch_sentences_length(batch) #num of tokens in each sentences in each docs [doc->#tokens_in_sent]
#    batch = torch.tensor(batch)
##    batch_idx_padded = pad_sentences(batch_idx, batch_sentences_len, pad_token) #makinbg every sent same lengths
##    padded_batch = Batch_Padding(batch_idx_padded, batch_first=True, pad_token=pad_token) #making every doc same length. 3D Tensor [doc->sentences->tokens]
##    modified_batch_sentences_len = pad_senteces_len(batch_sentences_len, padded_batch.size(1)) 
#
#    return batch, batch_docs_len, batch_sentences_len

def DocsToSent(docu_batch):
    sent_list = []
        
    doc_len = []
    sent_len = []

    for doc in docu_batch:
        doc_len.append(len(doc))

    max_doc_len = max(doc_len)
    pad_token = ["pad"]

    for doc in docu_batch:
        if len(doc)<max_doc_len:
            for _ in range(max_doc_len-len(doc)):
                doc.append(pad_token)
        for sent in doc:
            sent_len.append(len(sent))
            sent_list.append(sent)

    return sent_list, doc_len, np.asarray(sent_len)

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

class Batch_Generator_Global():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, dirname, filelist_path, args, test=False):
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
        self.filelist_path = filelist_path

        if self.test==True:
            self.batch_size = args.batch_size_test
        else:
            self.batch_size = args.batch_size

        self.shuffle = args.shuffle

    def __iter__(self):

        with open(self.filelist_path, 'r') as f:
            items = [line.strip() for line in f.readlines()]

        if self.shuffle == True:
            print("Shuffle the dataset")
            #random.seed(0) # fix the randomness
            random.shuffle(items)
        else:
            print("Without shuffle")

        batch = []
        batch_fname = []
        batch_length = []

        for i in range(20):
            for fname in items: 
                fname = fname + "_" + str(i+1)
                loadpath = os.path.join(self.dirname, fname)
                batch_file = Load_File(loadpath, self.file_types)

                if batch_file[0]==batch_file[1]:
                    continue
                for z in range(len(batch_file)):
                    batch_file[z] = [sentence.split() for sentence in batch_file[z]]
                batch.append(batch_file)
                batch_fname.append(fname)
                batch_length.append(len(batch_file[0]))

                if len(batch)==self.batch_size:
                    yield batch, batch_length, batch_fname
                    batch = [] # make it batch empty for the next iteration
                    batch_fname = [] 
                    batch_length = [] 


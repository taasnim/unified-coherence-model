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
from itertools import tee, chain
from operator import itemgetter

import gensim
#import logging_tool, DataPrep
from utilities import logging_tool, DataPrep

from torch.utils import data
from torch.utils.data import Dataset
import torch
import torch.nn as nn

def batch_eos_label(batch):
    batch_eos_position = [sent_eos_label(docu) for docu in batch]
    return batch_eos_position

def sent_eos_label(docu):
    eos_label = [1 if word=="</s>" else 0 for word in docu]
    eos_position = [pos for pos, eos in enumerate(eos_label) if eos==1]
    #return eos_label
    return eos_position

def Unpairing_Pos_Neg(doc_batch):
    pos_batch = []
    neg_batch = []
    for pos_neg in doc_batch:
        pos_batch.append(pos_neg[0])
        neg_batch.append(pos_neg[1])

    pos_eos_labels = batch_eos_label(pos_batch)
    neg_eos_labels = batch_eos_label(neg_batch)
    return pos_batch, neg_batch, pos_eos_labels, neg_eos_labels

def Label_Creator_Batch(doc_batch, device):
    pos_labels = []
    neg_labels = []
    for doc in doc_batch:
        pos_batch = doc[0]
        neg_batch = doc[1]
        '''
        ??? for same token in same position in pos sentence and neg sentence => neg token will be 1
        ??? for same length of sentences of pos and neg, eos token (</s>) in neg will be 1.
            We are using the eos token label as the sentence label. So whole sentence will be regarded as pos 
        '''

        label = torch.FloatTensor([int(pos==neg) for pos, neg in zip(pos_batch, neg_batch)])
        pos_label = torch.ones(label.size(), dtype=torch.float).to(device)
        neg_label = label.to(device)
        pos_labels.append(pos_label)
        neg_labels.append(neg_label)

    return pos_labels, neg_labels

def Label_Creator(pos_batch, neg_batch, device):
    label = torch.FloatTensor([int(pos==neg) for pos, neg in zip(pos_batch, neg_batch)])

    pos_label = torch.ones(label.size(), dtype=torch.float).to(device)
    neg_label = label.to(device)

    return pos_label, neg_label

def Previous_Current(previous, current):
    prev_pos = previous[0]
    prev_neg = previous[1]

    curr_pos = current[0]
    curr_neg = current[1]
    return prev_pos, prev_neg, curr_pos, curr_neg

def Load_On_Device(tensors, device):
    for tensor in tensors:
        yield tensor.to(device)

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def Dataset_Iterator(dataset):
    for data in dataset:
        yield data

def Dictionary_Info(dictionary):
    keys = []
    for key, value in dictionary.items():
        keys.append(key)
    return keys

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
        

def Save_File(savepath, data, types):
    if types=='pickle':
        with open(savepath, 'wb') as fp:
            pickle.dump(savepath, fp, protocol = pickle.HIGHEST_PROTOCOL)

    elif types=='json':
        with open(savepath, 'w') as fout:
            json.dump(data, fout)

    elif types=='npy':
        data = np.save(savepath, data)
        return data
    else:
        print("json or pickle only")


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

class Pos_Neg_Pairing():
    def __init__(self, pos_path, neg_dir_path, window, text):
        self.pos_path = pos_path
        self.neg_dir_path = neg_dir_path
        self.window = window
        self.text = text

    def Load_Pos_Names(self):
        """
        Search pos_files in a pos_folder within the path
        """
        dir_paths = os.walk(self.pos_path) # Return Generator
        if self.text==True:
            pattern = r"^wsj+_\d+\.pos\.text$"
            regex = re.compile(pattern)

            # Enter subfolders. ex) 01, 02, 03....
            File_List = []
            for dir_path in dir_paths:
                root_dir = dir_path[0]
                file_path_list = dir_path[2]

                # Read only text files
                for file_name in file_path_list:
                    if regex.match(file_name) is not None:
                        File_List.append(file_name)

            return File_List
        else:
            pattern = r"^wsj+_\d+\.pos\.text.+$"
            regex = re.compile(pattern)

            # Enter subfolders. ex) 01, 02, 03....
            File_List = []
            for dir_path in dir_paths:
                root_dir = dir_path[0]
                file_path_list = dir_path[2]

                # Read only text files
                for file_name in file_path_list:
                    if regex.match(file_name) is not None:
                        File_List.append(file_name)

            return File_List

    def Load_Neg_Names(self, pos_file_name):
        """
        Take all negative documents of a positive document

        window: {0, 1, 2, 3}
        - 0: take all windows
        """

        #pos_filename = str(pos_file_name[0]) # Torch loader...
        pos_filename = str(pos_file_name)
        if self.text==True:
            if self.window==0:
                pattern = re.escape(pos_filename)+r".+$"
            elif self.window!=0:
                pattern = re.escape(pos_filename)+r"."+re.escape(str(self.window))+r".*$"
            else:
                print("window should be \{0, 1, 2, 3\}")
        else:
            if self.window==0:
                pattern = re.escape(pos_filename)+r".*$"
            elif self.window!=0:
                pattern = re.escape(pos_filename)+r"."+re.escape(str(self.window))+r".*$"
            else:
                print("window should be \{0, 1, 2, 3\}")

        regex = re.compile(pattern)
        dir_paths = os.walk(self.neg_dir_path) # Return Generator
        Neg_File_List = []
        for dir_path in dir_paths:
            root_dir = dir_path[0]
            file_path_list = dir_path[2]

            # Read only text files
            for file_name in file_path_list:
                if regex.match(file_name) is not None:
                    Neg_File_List.append(file_name)

        return Neg_File_List
        
def Create_Path(dir_path, filename):
    created_path = os.path.join(dir_path, filename)
    return created_path

def Open_Doc(dir_path, filename):
    filepath = os.path.join(dir_path, filename)
    with open(filepath, encoding='utf-8') as fp:
        text = fp.read().splitlines()
    return text

def Doc_Generator(dir_path, pos_file_names):
    for filename in pos_file_names:
        filepath = os.path.join(dir_path, filename)
        with open(filepath, encoding='utf-8') as fp:
            text = fp.read().splitlines()
        yield text, filename

def Doc_Size(doc, types='text'):
    if types=='text':
        doc_length = len(doc)
        return doc_length

    else:
        sent_0 = doc[0].split(" ")
        doc_length = len(sent_0[1:])
        return doc_length


def Batch2Idx(batch, word2idx):
    """
    - Input should be a list of list with tokens
        ex) [["he", "is", "good"], ["she", "is", "good"]]
    - Iput could be a batch or whole document
    """
    batch_idx = []
    for sent in batch:
        sent_list = []
        for word in sent:
            try:
                idx = word2idx[word]
                sent_list.append(idx)
            except:
                idx = word2idx['<unk>']
                sent_list.append(idx)
        batch_idx.append(sent_list)
    return batch_idx

def Batch_Length_Calculator(batch, pad_token):
    """
    batch_len: Return a list of the length of sentences in a batch
    """
    batch_len = []
    valid_batch_len = []
    for sent in batch:
        if sent[0]==pad_token:
            batch_len.append(len(sent))
            valid_batch_len.append(0)
        else:
            batch_len.append(len(sent))
            valid_batch_len.append(len(sent))
    return batch_len, valid_batch_len

def Batch_Padding(batch, batch_first, pad_token):
    """
    Sentences that are shorter than the max sentence length will be padded
    """
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch], batch_first=True, padding_value=pad_token)
    return padded_batch

def Sort_List(lists):
    sorted_list_zipped = [(x[1], x[0]) for x in sorted(enumerate(lists), key=lambda x: len(x[1]), reverse=True)]
    sorted_list = [x[0] for x in sorted_list_zipped]
    sorting_idx = [x[1] for x in sorted_list_zipped]
    return sorted_list, sorting_idx

def zipped_list(sorted_list, sorting_idx):
    return list(zip(sorted_list, sorting_idx))

def Unsort(zipped_list):
    unsorted = torch.stack([x[0] for x in sorted(zipped_list, key=itemgetter(1))])
    return unsorted

def Batch_Size_Normalization(batch, batch_len, valid_batch_len, sorting_idx, pad_token, batch_size):
    """
    To maintain the minibatch size as the predefined size, 
    a minibatch with a smaller size will be padded with some padding value

    batch_len: length of sentences in a mini-batch
    batch_size: mini-batch size
    """
    max_length = max(batch_len)
    current_batch_len = len(batch)
    need_more = batch_size-current_batch_len
    if need_more==0:
        return batch

    padding_array = np.ones(max_length)*pad_token
    for i in range(need_more):
        batch.append(padding_array)
        batch_len.append(1)
        valid_batch_len.append(0)
        sorting_idx.append(len(sorting_idx))
    return batch

def Concatenate_Tensor(data, dim):
    return torch.cat(data, dim=dim)

def Output_Masking(output, output_masking, device):
    masked_output = output
    for i, mask in enumerate(output_masking):
        if mask==0:
            mask_ = torch.ones_like(masked_output).to(device).requires_grad_(False)
            mask_[i] = mask_[i]*0
            masked_output = masked_output*mask_
    return masked_output

def score_masking(scores, output_masking, args):
    score_squeezed = scores.squeeze()
    score_mask = torch.tensor(output_masking, dtype=torch.float, device=args.device)
    return score_squeezed*score_mask

def Batch_Preprocessing(batch, word2idx, batch_size, batch_first=True, pad_token="<pad>"):
    """
    1) Batch is the raw words (not indice)
    2) Batch should be at the first dim
    3) Batch should be sorted in the descending order to use the "packed padded sequence function"

    - batch_size: size of mini-batch
    """
    pad_token = word2idx[pad_token]
    batch_idx = Batch2Idx(batch, word2idx)

    batch_idx, sorting_idx = Sort_List(batch_idx)

    batch_len, valid_batch_len = Batch_Length_Calculator(batch_idx, pad_token)
    batch_idx = Batch_Size_Normalization(batch_idx, batch_len, valid_batch_len, sorting_idx, pad_token, batch_size=batch_size)
    padded_batch = Batch_Padding(batch_idx, batch_first=batch_first, pad_token=pad_token)
    return padded_batch, batch_len, valid_batch_len, sorting_idx

def Docu_Size_Normalization(batch, batch_len, pad_token, batch_size):
    """
    Make documents in a batch have the same length
    """
    padded_batch = []
    max_doc_len = max(batch_len)
    need_more_list = []

    for doc in batch:
        pos_doc = doc[0]
        neg_doc = doc[1]

        need_more = max_doc_len-len(pos_doc)
        if need_more==0:
            continue
        else:
            padding_array = ['<pad>']
            need_more_list.append(need_more)
            for i in range(need_more):
                pos_doc.append(padding_array)
                neg_doc.append(padding_array)

    return batch

def Document_Masking(concat_hidden, batch_len, valid_batch_len, device):
    output_masking = [1]*len(batch_len)
    Doc_Batch_Length = [1]*len(batch_len)

    masked_hidden = concat_hidden
    for i, length in enumerate(zip(batch_len, valid_batch_len)):
        if length[0]!=length[1]:
            mask_ = torch.ones_like(concat_hidden).to(device).requires_grad_(False)
            mask_[i] = mask_[i]*0
            masked_hidden = masked_hidden*mask_
            output_masking[i] = 0
    
    return masked_hidden, output_masking, Doc_Batch_Length

def Concatenate_Tensor(data, dim):
    return torch.cat(data, dim=dim)

def Sentence_Emb_Processing(hidden_state, batch_len, valid_batch_len, sorting_idx, device):
    concat_hidden = Concatenate_Tensor((hidden_state[0], hidden_state[1]), dim=1)
    concat_hidden, output_masking, Doc_Batch_Length = Document_Masking(concat_hidden, batch_len, valid_batch_len, device)
    zipped = zipped_list(concat_hidden, sorting_idx)
    unsorted_sent_embedding = Unsort(zipped)
    unsorted_sent_embedding = unsorted_sent_embedding.unsqueeze(1)
    return unsorted_sent_embedding, output_masking, Doc_Batch_Length

def Doc_LSTM_Processing(output, hidden_dim, output_masking, sorting_idx, args):
    concat_output = Concatenate_Tensor((output[:,:,:hidden_dim], output[:,:,hidden_dim:]), dim=2)
    masked_output = Output_Masking(concat_output, output_masking, args.device)
    return masked_output
    
def Batch_Preprocessing_Pos_Neg(pos_batch, neg_batch, word2idx, batch_size, batch_first=True, pad_token="<pad>"):
    pos_sent_batch, pos_batch_len = Batch_Preprocessing(pos_batch, word2idx, batch_size, batch_first=True, pad_token="<pad>")
    neg_sent_batch, neg_batch_len = Batch_Preprocessing(neg_batch, word2idx, batch_size, batch_first=True, pad_token="<pad>")
    return pos_sent_batch, pos_batch_len, neg_sent_batch, neg_batch_len


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

        if test==True:
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

#class Sentence_Batching_Unpaired():
#    """
#    - get a document and return sentences
#    """
#    def __init__(self, doc, label, eos, batch_len, pad_idx, args, data_name):
#        self.doc = doc
#        self.n_docs = batch_len
#        self.batch_size = args.batch_size
#        self.data_name = data_name
#        self.label = label
#        self.eos = eos
#        
#    def __iter__(self):
#
#        max_doc_len = max(self.n_docs)
#        
#        sent_batch = []
#        labels = []
#        eos_list = []
#        for s_i in range(max_doc_len):
#            try: 
#                sent_batch = [docu[s_i] for docu in self.doc]
#                labels = [lab[s_i] for lab in self.label]
#                eos_list = [_eos[s_i] for _eos in self.eos]
#            except:
#                print(f"dataname: {self.data_name}") 
#            
#            if len(sent_batch)==self.batch_size:
#                yield (sent_batch, labels, eos_list)
#                sent_batch = []
#                labels = []
#                eos_list = []
#
#        if sent_batch: 
#            yield (sent_batch, labels, eos_list)

#class Sentence_Batching_Unpaired():
#    """
#    - get a document and return sentences
#    """
#    def __init__(self, doc, label, eos, batch_len, pad_idx, args, data_name):
#        self.doc = doc
#        self.n_docs = batch_len
#        self.batch_size = args.batch_size
#        self.data_name = data_name
#        self.label = label
#        self.eos = eos
#        
#    def __iter__(self):
#
#        max_doc_len = max(self.n_docs)
#        
#        sent_batch = []
#        labels = []
#        eos_list = []
#        for s_i in range(max_doc_len):
#            try: 
#                sent_batch = [docu[s_i] for docu in self.doc]
#                labels = [lab[s_i] for lab in self.label]
#                eos_list = [_eos[s_i] for _eos in self.eos]
#            except:
#                print(f"dataname: {self.data_name}") 
#            
#            if len(sent_batch)==self.batch_size:
#                yield (sent_batch, labels, eos_list)
#                sent_batch = []
#                labels = []
#                eos_list = []
#
#        if sent_batch: 
#            yield (sent_batch, labels, eos_list)


class Sentence_Batching():
    """
    - get a document and return sentences
    - each doc consists of pos and neg documents
    """
    def __init__(self, docs, batch_len, pad_idx, args, data_name):
        self.docs = docs
        self.n_docs = batch_len
        self.batch_size = args.batch_size
        self.data_name = data_name
        
    def __iter__(self):

        max_doc_len = max(self.n_docs)
        

        pos_sent_batch = []
        neg_sent_batch = []
        for s_i in range(max_doc_len):
            try: 
                pos_sent_batch = [doc[0][s_i] for doc in self.docs]
                neg_sent_batch = [doc[1][s_i] for doc in self.docs]
            except:
                print(f"dataname: {self.data_name}") 
            
            if len(pos_sent_batch)==self.batch_size:
                yield pos_sent_batch, neg_sent_batch 
                pos_sent_batch = []
                neg_sent_batch = []

        if pos_sent_batch: 
            yield pos_sent_batch, neg_sent_batch 

def Permutation_Checker(pos, neg):
    i=0
    for sent_1, sent_0 in zip(pos, neg):
        print(f"Sentence {i}: {sent_1==sent_0}") 
        i+=1

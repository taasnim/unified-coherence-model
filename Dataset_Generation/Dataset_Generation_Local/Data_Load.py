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

import gensim
import logging_tool

from torch.utils import data
from torch.utils.data import Dataset
import torch
import torch.nn as nn


def Dataset_Iterator(dataset):
    for data in dataset:
        yield data

def Dictionary_Info(dictionary):
    keys = []
    for key, value in dictionary.items():
        keys.append(key)
    return keys

#def Keras_Padding(X_1, X_0, maxlen=25000):
#    X_1 = sequence.pad_sequences(X_1, maxlen)
#    X_0 = sequence.pad_sequences(X_0, maxlen)
#    return X_1, X_0

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
            text = fp.read().strip().splitlines()
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
#    batch_idx = []
#    for sent in batch:
#        sent_list = []
#        for word in sent:
#            try:
#                idx = word2idx[word]
#                sent_list.append(idx)
#            except:
#                idx = word2idx['<unk>']
#                sent_list.append(idx)
#        batch_idx.append(sent_list)
    batch_idx = [[word2idx[word] for word in sent]for sent in batch]
    return batch_idx

def Batch_Length_Calculator(batch):
    """
    batch_len: Return a list of the length of sentences in a batch
    """
    batch_len = [len(sent) for sent in batch]
    return batch_len

def Batch_Padding(batch, batch_first, pad_token):
    """
    Sentences that are shorter than the max sentence length will be padded
    """
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in batch], batch_first=True, padding_value=pad_token)
    return padded_batch

def Batch_Size_Normalization(batch, batch_len, pad_token, batch_size):
    """
    To maintain the minibatch size as the predefined size, 
    a minibatch of smaller size will be padded with some padding value

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
    return batch

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

    
def Batch_Preprocessing(batch, word2idx, batch_size, batch_first=True, pad_token="<pad>"):
    """
    Batch is the raw words (not indice)
    Batch should be at the first dim
    Batch should be sorted in the descending order to use the "packed padded sequence function"

    - batch_size: size of mini-batch
    """
    pad_token = word2idx[pad_token]
    batch_idx = Batch2Idx(batch, word2idx)

    batch_idx.sort(key = lambda s: len(s), reverse=True) # sort the batch in the descending order

    batch_len = Batch_Length_Calculator(batch_idx)
    batch_idx = Batch_Size_Normalization(batch_idx, batch_len, pad_token, batch_size)
    padded_batch = Batch_Padding(batch_idx, batch_first=batch_first, pad_token=pad_token)
    return padded_batch, batch_len

class Batch_Generator():
    """
    1) Read all text files in a folder
    2) Return documents!!
    >> Document batching!
    """
    def __init__(self, dirname, args):
        """
        - dirname: file directory name
        - file_types: json or pickle...
        - batch_size: mini batch size
        - shuffle: shuffle the dataset?
        - batch_length = length of each document in a mini-batch
        """
        self.dirname = dirname
        self.file_types = args.file_types
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

        # this is equivalent to "if len(batch)!=0:"
        # flush all the remaining batch 
#        if batch: 
#            yield batch, batch_length, batch_fname 

class Sentence_Batching():
    """
    - Get a document and return sentences
    - Each doc consists of pos and neg documents
    """
    def __init__(self, docs, batch_len, pad_idx, args):
        self.docs = docs
        self.n_docs = batch_len
        self.batch_size = args.batch_size
        
    def __iter__(self):

        max_doc_len = max(self.n_docs)
        
        pos_sent_batch = []
        neg_sent_batch = []
        for S_i in range(max_doc_len):
            try: 
                pos_sent_batch = [doc[0][S_i] for doc in self.docs]
                neg_sent_batch = [doc[1][S_i] for doc in self.docs]
            except:
                length = [len(doc[0]) for doc in self.docs]
                print(f"max_doc_len: {max_doc_len}") 
                print(f"doc_length: {length}") 
                print(f"S_i: {S_i}") 
                pdb.set_trace()  
            
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

def Parse_Args():

    parser = argparse.ArgumentParser()

    # Experiment Setting
    parser.add_argument('--local_window', type=int, default=3, help='Number of local window')
    parser.add_argument('--model_save_dir', type=str, default="/home/han/Desktop/coherence_pytorch/saved_model/", help='Number of local window')
    parser.add_argument('--resume_train', type=bool, default=False, help='Load saved model and resume training')

    # Dataset parameter
    parser.add_argument('--n_window', type=int, default=1, help='Number of permutation window')
    parser.add_argument('--train_path', type=str, default="/home/han/Desktop/sample_test/train_egrid_perm/", help='Save train paired Data') # Test purpose
    #parser.add_argument('--train_path', type=str, default="/home/han/Desktop/Dataset/train/", help='Save train paired Data')
    parser.add_argument('--test_path', type=str, default="/home/han/Desktop/Dataset/test/", help='Save test paired Data')
    parser.add_argument('--pre_embedding_path', type=str, default="./Pretrained_Embedding/GoogleNews-vectors-negative300.bin", help='Pretrained word embedding path')
    parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension')

    # Vocab arguments
    parser.add_argument('--vocab_path', type=str, default="/home/han/Desktop/Dataset/vocab/Vocab", help='Vocab path') 
    parser.add_argument('--word2idx_path', type=str, default="/home/han/Desktop/Dataset/vocab/word2idx", help='word2idx')
    parser.add_argument('--idx2word_path', type=str, default="/home/han/Desktop/Dataset/vocab/idx2word", help='idx2word')

    # Training Parameter-------------------------------------------------------------
    parser.add_argument('--Epoch', type=int, default=1, help='Number of Epoch ')
    parser.add_argument('--adaptive_learning_rate', type=int, default=15, help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate')
    parser.add_argument('--ranking_loss_margin', type=float, default=0.1, help='ranking loss margin')
    parser.add_argument('--device', type=str, default='cuda', help='CPU? GPU?')

    # Minibatch argument
    parser.add_argument('--batch_size', type=int, default=3, help='Mini batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle items')
    parser.add_argument('--file_types', type=str, default='json', help='Load file type')
    parser.add_argument('--window_size', type=int, default=3, help='Local window size')
    #parser.add_argument('--seed', type=str, default='json', help='Load file type')

    # Network Parameter
    parser.add_argument('--n_vocabs', type=int, help='Word embedding dim, it should be defined using the vocab list')
    parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden dim of RNN')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio of RNN')
    parser.add_argument('--pretrained_embed', type=bool, default=True, help='Use pretrained Word embedding?')
    parser.add_argument('--bidirectional', type=bool, default=True, help='Bi-directional RNN?')
    parser.add_argument('--batch_first', type=bool, default=True, help='Dimension order')

    return parser.parse_args()

if __name__ == "__main__":

    now = datetime.datetime.now()
    oov_logger = logging_tool.Setup_Logger('OOV_info', '/home/han/oov_info.log')
    performance_logger = logging_tool.Setup_Logger('train_info', '/home/han/train_info.log')
    loss_logger = logging_tool.Setup_Logger('loss_info', '/home/han/loss_info.log')

    args = Parse_Args()

    # Device Setting
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        current_gpu = torch.cuda.current_device()
        gpu_name = cuda.Device(current_gpu).name()
        print(f"Running on GPU: {gpu_name}") 
        #torch.cuda.manual_seed(0)
    else:
        args.device = torch.device('cpu')
        print("Running on CPU")
        #torch.manual_seed(0)


    vocab = Load_File(args.vocab_path, types='json')
    args.n_vocabs = len(vocab)

    word2idx = Load_File(args.word2idx_path, types='json')
    idx2word = Load_File(args.idx2word_path, types='json')

    local_model = Local_Coherence(args, vocab, word2idx, idx2word, emb_trainable=True, pad_idx=word2idx['<pad>'], oov_logger=oov_logger)
    local_model = local_model.to(args.device) # Load on GPU or CPU

    args.resume_train = True
    if args.resume_train==True:
        model_state = 'Epoch_0_Mini_Batch_2_MMdd_2_13'
        model_load_path = os.path.join(args.model_save_dir, model_state)
        local_model.load_state_dict(torch.load(model_load_path))
        local_model = local_model.to(args.device) # Load on GPU or CPU

    optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
    criterion = nn.MarginRankingLoss(margin=args.ranking_loss_margin)

    batch_generator = Batch_Generator(args.train_path, args)
    Best_Result = 0
    Loss = []
    for epoch in range(args.Epoch):
        print(f"Current Epoch: {epoch}-------------------------------------------------") 

        # Learning rate will be decreased for every certain epochs
        Adaptive_Learning_Rate(optimizer, epoch, args.learning_rate, period=args.adaptive_learning_rate)

        N_Data = 0
        N_TP = 0
        for n_mini_batch, (batch, batch_sent_len, data_name) in enumerate(batch_generator):
            """
            Batch the document
            """
            max_sent_len = max(batch_sent_len)

            doc_batch = Docu_Size_Normalization(batch, batch_sent_len, word2idx['<pad>'], args.batch_size)

            # For each document, initial prev_hidden is set to zero values
            pos_prev_hidden = torch.zeros(args.batch_size, args.hidden_dim*2) # *2: concatenated hidden dim
            neg_prev_hidden = torch.zeros(args.batch_size, args.hidden_dim*2)

            sent_batching = Sentence_Batching(doc_batch, batch_sent_len, word2idx['<pad>'], args)

            pos_biaffine_window = []
            neg_biaffine_window = []

            pos_local_score = []
            neg_local_score = []

            for n_sent, (pos_sent_batch, neg_sent_batch) in enumerate(sent_batching):
                """
                Batch the sentence in the documents
                """

                pos_sent_batch, pos_batch_len = Batch_Preprocessing(pos_sent_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
                neg_sent_batch, neg_batch_len = Batch_Preprocessing(neg_sent_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")

                pos_sent_batch = pos_sent_batch.to(args.device)
                neg_sent_batch = neg_sent_batch.to(args.device)

                (pos_hidden, pos_biaffine), (neg_hidden, neg_biaffine) = local_model.Forward_Siamese(pos_sent_batch, neg_sent_batch, 
                        pos_batch_len, neg_batch_len, pos_prev_hidden, neg_prev_hidden)

                pos_prev_hidden = pos_hidden
                neg_prev_hidden = neg_hidden

                pos_biaffine_window.append(pos_biaffine)
                neg_biaffine_window.append(neg_biaffine)

                if len(pos_biaffine_window)==2: # len(pos_biaffine_window) shoule be two, because it is the biaffine 

                    pos_window = torch.cat(pos_biaffine_window, dim=1)
                    neg_window = torch.cat(neg_biaffine_window, dim=1)

                    optimizer.zero_grad()

                    pos_score, neg_score = local_model.Coherence_Socre(pos_window, neg_window)
                    pos_local_score.append(pos_score.data)
                    neg_local_score.append(neg_score.data)

                    label = torch.ones(pos_score.size(0))
                    loss = criterion(pos_score, neg_score, label)
                    loss.backward(retain_graph=True)

                    optimizer.step()

                    # make biaffine window free
                    if max_sent_len%2==1 and max_sent_len-n_sent==2:
                        pos_remain = [pos_biaffine_window[-1]]
                        neg_remain = [neg_biaffine_window[-1]]

                    pos_biaffine_window = []
                    neg_biaffine_window = []


                elif n_sent+1==max_sent_len:
                    pos_remain.append(pos_biaffine)
                    neg_remain.append(neg_biaffine)

                    pos_window = torch.cat(pos_remain, dim=1)
                    neg_window = torch.cat(neg_remain, dim=1)

                    optimizer.zero_grad()

                    pos_score, neg_score = local_model.Coherence_Socre(pos_window, neg_window)
                    pos_local_score.append(pos_score.data)
                    neg_local_score.append(neg_score.data)

                    label = torch.ones(pos_score.size(0))
                    loss = criterion(pos_score, neg_score, label)
                    loss.backward(retain_graph=True)

                    optimizer.step()

            Pos_Doc_Score = np.asarray([score.data.numpy() for score in pos_local_score]).sum(axis=0)
            Neg_Doc_Score = np.asarray([score.data.numpy() for score in neg_local_score]).sum(axis=0)

            Score_Comparison = Pos_Doc_Score>Neg_Doc_Score 
            Score_Comparison = Score_Comparison*1 # True->1, False->0

            N_Data += len(Score_Comparison)

            N_Correct = Score_Comparison.sum()
            N_TP+=N_Correct
            Loss.append(loss)
            loss_logger.info(f"Mini-Batch No.: {n_mini_batch} || Batch Accuracy: {N_Correct/args.batch_size}|| Loss: {loss}")
            print(f"Score: {Score_Comparison}") 
            print(f"N_Correct: {N_Correct}") 
            print(f"Mini-Batch No.: {n_mini_batch} || Batch Accuracy: {N_Correct/args.batch_size}||Loss: {loss:.7f}") 

        acc = N_TP/N_Data
        if Best_Result<acc:
            Best_Result=acc
            model_name = f"Epoch_{epoch}_Mini_Batch_{n_mini_batch}_MMdd_{now.month}_{now.day}"
            model_save_path = os.path.join(args.model_save_dir, model_name)
            Save_Model_State(local_model, model_save_path)


        print(f"Training Epoch: {epoch}|| Mini-Batch No.: {n_mini_batch}|| accuracy result: {acc}|| Best result so far: {Best_Result}||") 
        performance_logger.info(f"Training Epoch: {epoch}|| Mini-Batch No.: {n_mini_batch}|| accuracy result: {acc}|| Best result so far: {Best_Result}||")



                    

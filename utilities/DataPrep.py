# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Filename: DataPrep.py 
# Developed: Hancheol Moon
# Purpose: Save paired data (pos+neg)
# ----------------------------------------------------------------
# ----------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
import torchtext
import spacy
import json
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
import os, sys
import operator
import nltk
from nltk import sent_tokenize
import copy

import argparse 
import pdb
import logging
import re

#import Data_Load 
from utilities import Data_Load 
import datetime



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

def wsj_tokenizer(text):
    """
    This is a simple tokinizer
    Tokenize based on "space"
    
    Remove <s>, </s>
    """
    tokenized = [sent.strip().split()[1:-1] for sent in text]
    return tokenized

def Simple_Tokenizer(document):
    """
    - Tokenize using simple_preprocess in Gensim library
    - Return tokenized documents
    - Each list element is a sentence
    """
    tokenized = [simple_preprocess(sent, deacc=True) for sent in document]

    return tokenized

class ReadTxtFiles():
    """
    1) Read all text files in a folder
    2) Load a setence by sentence
    3) tokenize using simple_preprocess
    """
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='latin'):
                line = Sentence_Cleaner(line)
                yield line

def NormalizeString(s):
    if type(s)==list:
        s = " ".join(s)
    s = s.lower().strip()
    s = re.sub(r"(<s>|</s>)", r"", s) # remove <s>, </s>
    s = re.sub(r"('s|'d|'ll|'t|'n)", r"", s) # Remove contraction

    s = re.sub(r"([\d]+s)", r"", s) # remove words like '1950s'

    s = re.sub(r"([.!?])", r" ", s) 
    s = re.sub(r"[^a-zA-Z.!?@]+", r" ", s)

#    s = re.sub(r"[^a-zA-Z]+", r" ", s)
#    s = re.sub(r"\d+", r"", s)
    return s

def StopWords(sent):
    sent = sent.split()
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", 
            "he", "him", "his", "himself", "she", "her", "hers", "herself", 
            "it", "its", "itself", 
            "they", "them", "their", "theirs", "themselves", 
            "what", "which", "who", "whom", 
            "this", "that", "these", "those", 
            "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", 
            "do", "does", "did", "doing", 
            "a", "an", "the", 
            "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
            "again", "further", "then", "once", "here", "there", 
            "when", "where", "why", "how", 
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
            "no", "nor", "not", "only", "own", "same", "so", 
            "than", "too", "very", "can", "will", "just", "don", "should", "now",
            "n"]

    removed_sent = copy.deepcopy(sent)
    for word in sent:
        if word in stopwords:
            removed_sent.remove(word)

    return removed_sent

def Sentence_Cleaner(sent):
    """
    Clean a sentence
    """
    sent = NormalizeString(sent)
    sent = StopWords(sent)
    return sent

def Document_Normalizer(doc):
    """
    Clean a document
    """
    normalized_doc = []
    for sent in doc:
        sent = Sentence_Cleaner(sent)
        if len(sent) ==0:
            continue
        normalized_doc.append(sent)
    return normalized_doc

def Word_Counter(file_path):
    """
    Count whole words in the path
    """
    word_count = {}
    for line in ReadTxtFiles(file_path):
        for word in line:
            if word not in word_count:
                word_count[word]=1
            else:
                word_count[word]+=1
    
    return word_count

def Vocab(doc, vocab, word2idx, idx2word):
    for sent in doc:
        for word in sent:
            if word not in vocab:
                vocab.append(word)

    for i, word in enumerate(vocab):
        if word not in word2idx:
            word2idx[word] = int(i)
            idx2word[int(i)] = word

    return vocab, word2idx, idx2word


def Build_Vocab(word_count, ratio=0.8):
    """
    Build vocabulary dictionary
    """
    n_vocab = len(word_count)
    word_count = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True) # Sort by value

    # Take N frequent vocabs only
    Vocab = word_count[:int(n_vocab*ratio)] 
    Vocab = [tup[0] for tup in word_count[:int(n_vocab*ratio)]]    

    # Append special tokens
    Vocab.append('<unk>')
    Vocab.append('<pad>')

    print(f"Take {len(Vocab)} words from {n_vocab}") 

    return Vocab

def Word_Dictionary(vocab):
    word2idx = {}
    idx2word = {}

    for i, word in enumerate(vocab):
        if word not in word2idx:
            word2idx[word] = int(i)
            idx2word[int(i)] = word

    return word2idx, idx2word

def Sent2Idx(word2idx, sent, Torch=True):
    Indice = [word2idx[word] for word in sent]
    if Torch==True:
        Indice = torch.LongTensor(Indice)
        return Indice
    elif Torch==False:
        return Indice
    else:
        print("Torch is True or False?")


def Parse_Args():
    parser = argparse.ArgumentParser()

    # Dataset parameter
    parser.add_argument('--n_window', type=int, default=3, help='Number of permutation window')
    parser.add_argument('--train_pos', type=str, default="/home/han/Desktop/text_dataset/training/", help='Pos Train file dir path')
    parser.add_argument('--train_neg', type=str, default="/home/han/Desktop/text_dataset/training_perm/", help='Neg Train file dir path')
    parser.add_argument('--test_pos', type=str, default="/home/han/Desktop/text_dataset/test/", help='Pos Test file dir path')
    parser.add_argument('--test_neg', type=str, default="/home/han/Desktop/text_dataset/test_perm/", help='Neg Test file dir path')
    parser.add_argument('--pre_embedding_path', type=str, default="./Pretrained_Embedding/GoogleNews-vectors-negative300.bin", help='Pretrained word embedding path')

    parser.add_argument('--train_save_path', type=str, default="/home/han/Desktop/Dataset/train/", help='Save train paired Data')
    parser.add_argument('--test_save_path', type=str, default="/home/han/Desktop/Dataset/test/", help='Save test paired Data')
    parser.add_argument('--vocab_save_path', type=str, default="/home/han/Desktop/Dataset/vocab/", help='Save paired Data')

    return parser.parse_args()

if __name__ =="__main__":
    logger = logging.getLogger('PyTorch Network')

    f_handler = logging.FileHandler('/home/han/PyTorch.log')
    f_handler.setLevel(logging.ERROR)

    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    args = Parse_Args()
    Print_Args(args)

    # Train Files
    Paired_Files = Data_Load.Pos_Neg_Pairing(args.train_pos, args.train_neg, window=args.n_window, text=True)
    pos_file_names = Paired_Files.Load_Pos_Names()
    Pos_Generator = Data_Load.Doc_Generator(args.train_pos, pos_file_names)

#    word_count = Word_Counter(args.train_pos)
#    Vocab = Build_Vocab(word_count, ratio=1.0)
#    word2idx, idx2word = Word_Dictionary(Vocab)
#
#    savepath = Data_Load.Create_Path(args.vocab_save_path, 'Vocab')
#    Data_Load.Save_File(savepath, Vocab, types='json')
#
#    savepath = Data_Load.Create_Path(args.vocab_save_path, 'word2idx')
#    Data_Load.Save_File(savepath, word2idx, types='json')
#
#    savepath = Data_Load.Create_Path(args.vocab_save_path, 'idx2word')
#    Data_Load.Save_File(savepath, idx2word, types='json')


    for pos_doc, pos_filename in Pos_Generator:
        doc_len = Data_Load.Doc_Size(pos_doc, types='Egrid')
        neg_file_list = Paired_Files.Load_Neg_Names(pos_filename)
        Neg_Generator = Data_Load.Doc_Generator(args.train_neg, neg_file_list)

        pos_doc = Document_Normalizer(pos_doc)

        for neg_doc, neg_filename in Neg_Generator:
            neg_doc = Document_Normalizer(neg_doc)

            doc_pair = (pos_doc, neg_doc)

            savepath = Data_Load.Create_Path(args.train_save_path, neg_filename)
            Data_Load.Save_File(savepath, doc_pair, types='json')
        


    # Test Files
    Paired_Files = Data_Load.Pos_Neg_Pairing(args.test_pos, args.test_neg, window=args.n_window, text=True)
    pos_file_names = Paired_Files.Load_Pos_Names()
    Pos_Generator = Data_Load.Doc_Generator(args.test_pos, pos_file_names)

    for pos_doc, pos_filename in Pos_Generator:
        doc_len = Data_Load.Doc_Size(pos_doc, types='Egrid')
        neg_file_list = Paired_Files.Load_Neg_Names(pos_filename)
        Neg_Generator = Data_Load.Doc_Generator(args.test_neg, neg_file_list)

        pos_doc = Document_Normalizer(pos_doc)

        for neg_doc, neg_filename in Neg_Generator:
            neg_doc = Document_Normalizer(neg_doc)

            doc_pair = (pos_doc, neg_doc)

            savepath = Data_Load.Create_Path(args.test_save_path, neg_filename)
            Data_Load.Save_File(savepath, doc_pair, types='json')

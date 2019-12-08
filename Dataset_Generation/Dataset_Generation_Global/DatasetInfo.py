"""
To-Do-List:
    I think the number of windows should be determined by the length of document
    Some documents should be longer than others and I think it should be considered...
"""
import numpy as np
import copy
import pdb
import os, sys
import random
import re
import pickle

def Print_by_Lines(Document):
    for sent in Document:
        print(sent)

def Save_Document(save_path, Document):
    with open(save_path, 'wb') as fp:
        pickle.dump(Document, fp)

def Load_Document(load_path):
    with open(load_path, 'rb') as fp:
        Document = pickle.load(fp)
    return Document

def Window_Counter(files, types, perm=True):
    Sum = 0
    if types=='text':
        if perm==True:
            for i in range(3):
                pattern = r"^wsj+_\d+\.pos\.text."+re.escape(str(i+1))+r".*$"
                regex = re.compile(pattern)

                count = 0
                for file_name in files:
                    count += bool(regex.findall(file_name))
                print(f"Window {i+1}:{count}")
                Sum+=count
        else:
            pattern = r"^wsj+_\d+\.pos\.text$"
            regex = re.compile(pattern)

            count = 0
            for file_name in files:
                count += bool(regex.findall(file_name))
            Sum+=count

    else:
        if perm==True:
            for i in range(3):
                pattern = r"^wsj+_\d+\.pos\.text\.parsed\.ner\.EGrid."+re.escape(str(i+1))+r".*$"
                regex = re.compile(pattern)

                count = 0
                for file_name in files:
                    count += bool(regex.findall(file_name))
                print(f"Window {i+1}:{count}")
                Sum+=count
        else:
            pattern = r"^wsj+_\d+\.pos\.text\.parsed\.ner\.EGrid$"
            regex = re.compile(pattern)

            count = 0
            for file_name in files:
                count += bool(regex.findall(file_name))
            Sum+=count

    print(f"Summation: {Sum}") 



def Dataset_Info(path, types, perm):
    print(path)
    files = os.listdir(path)
    Window_Counter(files, types, perm)

    
if __name__ =='__main__':

#    filepath_text = './training/wsj_0118.pos.text'
#    filepath_egrid = './training_egrid/wsj_0118.pos.text.parsed.ner.EGrid'
#
#    filepath_text_perm = './training_perm/wsj_0118.pos.text_3_5745'
#    filepath_egrid_perm = './training_perm_egrid/wsj_0118.pos.text.parsed.ner.EGrid_3_5745'

    filepath_text = './training/'
    filepath_egrid = './training_egrid/'

    filepath_text_perm = './training_perm/'
    filepath_egrid_perm = './training_perm_egrid/'

    Dataset_Info(filepath_text, 'text', False)
    Dataset_Info(filepath_egrid, 'egrid', False)

    Dataset_Info(filepath_text_perm, 'text', True)
    Dataset_Info(filepath_egrid_perm, 'egrid', True)

    print("-------------------------------------------")
    print("-------------------------------------------")
    print("-------------------------------------------")

    filepath_text = './test/'
    filepath_egrid = './test_egrid/'

    filepath_text_perm = './test_perm/'
    filepath_egrid_perm = './test_perm_egrid/'

    Dataset_Info(filepath_text, 'text', False)
    Dataset_Info(filepath_egrid, 'egrid', False)

    Dataset_Info(filepath_text_perm, 'text', True)
    Dataset_Info(filepath_egrid_perm, 'egrid', True)


#    print("START --------------------------------------")
#    print("--------------------------------------Text Positive")
#    Document = Load_Document(filepath_text)
#    Print_by_Lines(Document)
#    print("--------------------------------------")
#    print("--------------------------------------")
#
#
#
#    print("--------------------------------------Text Negative")
#    Document_0 = Load_Document(filepath_text_perm)
#    Print_by_Lines(Document_0)
#    print("--------------------------------------")
#    print(Document==Document_0)
#    print("--------------------------------------")
#    print("--------------------------------------")
#    print(f"Text Stat: {len(Document_0)}")
#
#    print("--------------------------------------EGrid Positive")
#    Document_1 = Load_Document(filepath_egrid)
#    Print_by_Lines(Document_1)
#    print("--------------------------------------")
#    print("--------------------------------------")
#
#    print("--------------------------------------EGrid Negative")
#    Document_2 = Load_Document(filepath_egrid_perm)
#    Print_by_Lines(Document_2)
#    print(Document_1==Document_2)
#    print("--------------------------------------")
#    print("--------------------------------------")
#    print(f"Egrid Stat: {len(Document_1)}")


import numpy as np
import pdb
import re
import pandas as pd
import sys, os
import pickle

import torchtext
from torchtext import data

def Print_by_Lines(Document):
    for sent in Document:
        print(sent)

def Save_Document(save_path, Document):
    with open(save_path, 'wb') as fp:
        pickle.dump(Document, fp)

def Load_Document(load_path):
    """Load Binary Pickle File"""
    with open(load_path, 'rb') as fp:
        Document = pickle.load(fp)
    return Document

def Dataset_Info(path, window):
    files = os.listdir(path)
    pattern = r"^wsj+_\d+\.pos\.text_"+str(window)+".*$"
    regex = re.compile(pattern)

    count = 0
    for file_name in files:
        count += bool(regex.findall(file_name))
    print(f"Window {window}:{count}")


def Create_Data_List(path):
    data_list = []
    return data_list

def Read_File_Path(path):
    """
    Search files in a folder within the path
    """

    dir_paths = os.walk(path) # Return Generator


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

def Read_All_Files(path):
    File_List = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        File_List = filenames
    return File_List, dirpath

def Create_Path(file_name, dirpath):
    path = os.path.join(dirpath, file_name)
    return path

def To_Text(file_name_list, dirpath, save_folder_path, file_type):
    if file_type=='text':
        for file_name in file_name_list:
            path = Create_Path(file_name, dirpath)
            save_path = Create_Path(file_name, save_folder_path)
            pos_text = Load_Document(path)

            with open(save_path, 'w') as f:
                for sent in pos_text:
                    f.write(sent)
    else:
        for file_name in file_name_list:
            path = Create_Path(file_name, dirpath)
            save_path = Create_Path(file_name, save_folder_path)
            pos_text = Load_Document(path)

            with open(save_path, 'w') as f:
                for sent in pos_text:
                    f.write(sent+'\n')


if __name__ =='__main__':
    path_test = '/home/han/Desktop/raw-wsj/test_egrid/'
    path_train = '/home/han/Desktop/raw-wsj/training_egrid/'

    save_path_test = '/home/han/Desktop/text_dataset/test_egrid/'
    save_path_train = '/home/han/Desktop/text_dataset/train_egrid/'

    File_List_test, dirpath_test = Read_All_Files(path_test)
    File_List_train, dirpath_train = Read_All_Files(path_train)

    To_Text(File_List_test, dirpath_test, save_folder_path = save_path_test, file_type='egrid')

    To_Text(File_List_train, dirpath_train, save_folder_path = save_path_train, file_type='egrid')


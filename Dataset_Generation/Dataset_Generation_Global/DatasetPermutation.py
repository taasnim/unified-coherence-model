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
import itertools
import argparse
from shutil import copyfile
import json



def Check_Dir(path):
    if not os.path.exists(path): # check directory 
        os.makedirs(path)

def Print_by_Lines(Document):
    for sent in Document:
        print(sent)

def Save_Document(save_path, Document, doc_type):
    if doc_type=='text':
        with open(save_path, 'w') as fp:
            for line in Document:
                fp.write(line+'\n')
    else:
        with open(save_path, 'w') as fp:
            for line in Document:
                fp.write(line+'\n')

def Load_Document(load_path):
    with open(load_path, 'rb') as fp:
        Document = pickle.load(fp)
    return Document

def Read_Document_List(path, n_sent, file_type):
    """
    1) Read_File_Path
    2) Read_Document
    3) Return training and test documents
    """
    File_Path = Read_File_Path(path, n_sent=n_sent, file_type=file_type)

    training_document = {}
    test_document = {}
    for file_path in File_Path:

        if file_type == 'text':
            section = int(file_path.split('/')[3]) 
            file_name = file_path.split('/')[4]
        else:
            file_name = file_path.split('/')[-1]
            section = int(file_name.split('.')[0][4:6])

        if section <=13:
            training_document[file_name]= Read_Document(file_path, file_type)
        else:
            test_document[file_name]= Read_Document(file_path, file_type)

    return training_document, test_document, File_Path

def Read_Document(path, file_type):
    with open(path, 'r') as f:
        if file_type == 'text':
            #Document = f.readlines()
            Document = [line.strip() for line in  f.readlines()]
            #Document = [line.strip() for line in  Document]
            return np.array(Document)
        else:
            Document = [line.strip() for line in  f.readlines()]
            return np.array(Document)

def Count_Lines(lines, n_sent, file_type, Text_Path):
    if file_type == 'text':
        if len(lines) >= n_sent:
            return True
        else:
            return False

    elif file_type == 'egrid':
        first_line = lines.split(' ')
        if len(first_line)-1 >= n_sent:
            return True
        else:
            return False
    else:
        print('File type error')

    
def Read_File_Path(path, n_sent, file_type):
    """
    file_type: 1) 'text' 2) 'egrid'

    Return text file path
    ex) ./raw/wsj/08/wsj_xxxx.pos.text
    """

    dir_paths = os.walk(path) # Return Generator

    if file_type == 'text':
        pattern = r"^wsj+_\d+\.pos\.text$"
        regex = re.compile(pattern)


        # Enter subfolders. ex) 01, 02, 03....
        Text_Path_List = []
        for dir_path in dir_paths:
            root_dir = dir_path[0]
            file_path_list = dir_path[2]

            # Read only text files
            for file_path in file_path_list:
                if regex.match(file_path) is not None:
                    Text_Path = os.path.join(root_dir, file_path)
                    with open(Text_Path, 'r') as f:
                        lines = f.readlines()
                        Exceed = Count_Lines(lines, n_sent, file_type, Text_Path)
                        if Exceed==True:
                            Text_Path_List.append(Text_Path)

        return Text_Path_List

    elif file_type =='egrid':
        pattern = r"^wsj+_+\d+\.+pos+\.+text+\.+parsed+\.+ner+\.+EGrid$"
        regex = re.compile(pattern)

        # Enter subfolders. ex) 01, 02, 03....
        Text_Path_List = []
        for dir_path in dir_paths:
            root_dir = dir_path[0]
            file_path_list = dir_path[2]

            # Read only text files
            for file_path in file_path_list:
                if regex.match(file_path) is not None:
                    Text_Path = os.path.join(root_dir, file_path)
                    with open(Text_Path, 'r') as f:
                        lines = f.readline().strip()
                        Exceed = Count_Lines(lines, n_sent, file_type, Text_Path)
                        if Exceed==True:
                            Text_Path_List.append(Text_Path)
        return Text_Path_List
    else:
        print('File type should be a text or egrid')

def File_Path_Checker(File_Path_Text, File_Path_EGrid):
    text_files = {0}
    egrid_files = {0}

    for text_file in File_Path_Text:
        text =text_file.split('/')[-1].split('.')[0]
        text_files.add(text)

    for egrid_file in File_Path_EGrid:
        egrid = egrid_file.split('/')[-1].split('.')[0]
        egrid_files.add(egrid)

    print(egrid_files.difference(text_files))
    
class Dataset_Processing():

    def __init__(self, path, n_sent, n_window, window_size, perm):
        self.path = path
        self.n_sent = n_sent
        self.n_window = n_window
        self.window_size = window_size
        self.perm = perm

    def Create_Dataset(self):
        """
        n_window: list of the number of windows
        dataset_type: 'train' or 'test'
        """
        training_document, test_document, File_Path_Text = Read_Document_List(read_path, n_sent=self.n_sent, file_type = 'text')
        training_document_egrid, test_document_egrid, File_Path_EGrid = Read_Document_List(read_path, n_sent=self.n_sent, file_type = 'egrid')

        self.Creating_Dataset(training_document, training_document_egrid, dataset_type='train')
        self.Creating_Dataset(test_document, test_document_egrid, dataset_type='test')
    
    def Creating_Dataset(self, Documents, Documents_Egrid, dataset_type):
        """
        - Documents are the dictionary type

        1) Save positive documents
        2) Create window starting indices
        3) Perform permutation
        4) Save negative documents
        """
        # processing training documents
        assert dataset_type.lower() == 'train' or dataset_type.lower() == 'test'
        
        if dataset_type.lower() == 'train':
            pos_dir = os.path.join(os.getcwd(), 'training')
            neg_dir = os.path.join(os.getcwd(), 'training_perm')

            pos_dir_egrid = os.path.join(os.getcwd(), 'training_egrid')
            neg_dir_egrid = os.path.join(os.getcwd(), 'training_perm_egrid')

        else:
            pos_dir = os.path.join(os.getcwd(), 'test')
            neg_dir = os.path.join(os.getcwd(), 'test_perm')

            pos_dir_egrid = os.path.join(os.getcwd(), 'test_egrid')
            neg_dir_egrid = os.path.join(os.getcwd(), 'test_perm_egrid')

        Check_Dir(pos_dir)
        Check_Dir(neg_dir)

        Check_Dir(pos_dir_egrid)
        Check_Dir(neg_dir_egrid)


        for file_name, text in Documents.items():
            print(f"Processing {file_name}") 
            
            Document = text
            text_name = file_name # Text file

            egrid_file = file_name + '.parsed.ner.EGrid'
            EGrid = Documents_Egrid[egrid_file]

            # Save positive text file
            save_path = os.path.join(pos_dir, text_name) # Define save path
            Save_Document(save_path, Document, doc_type='text') # save positive documents

            # Save positive grid file
            save_path = os.path.join(pos_dir_egrid, egrid_file) # Define save path
            Save_Document(save_path, EGrid, doc_type='egrid') # save positive egrid

            for n, p in zip(self.n_window, self.perm):
                # window_idx is the all possible window indices
                n_permutation = p
                window_idx  = Sentence_Window(Document, n, self.window_size, perm=p, overlap=False) 
                window_idx_len = len(window_idx)

                if window_idx_len>=20:
                    window_idx  = [window_idx[i] for i in range(n_permutation)]
                    window_idx_len = len(window_idx)


                last_perm = []

                check_n_window = 0
                for window in window_idx:
                    perm_idx = []
                    for idx_list in window:
                        perm_list = list(itertools.permutations(idx_list))[1:] # Exclude the same thing
                        comb_selection = np.random.choice(range(len(perm_list)), 1, replace=False) # Randomly selects some combination indices
                        perm_element = list(perm_list[comb_selection[0]])
                        perm_idx.append(perm_element)

                    #perm_idx = [list(np.random.permutation(idx_list)) for idx_list in window]

                    check_n_window +=1
                    i = check_n_window

                    Permuted_Document, permuted_idx_text = Permutation(Document, window, perm_idx, file_type='text')
                    
                    Permuted_EGrid, permuted_idx_egrid = Permutation(EGrid, window, perm_idx, file_type='egrid')

                    save_name = text_name + '_' + str(n) + '_' + str(i)
                    save_path = os.path.join(neg_dir, save_name)
                    if len(Permuted_Document)!=len(Document):
                        print(f"Errorneous File: {save_name}") 
                    Save_Document(save_path, Permuted_Document, doc_type='text') # save negative documents

                    save_name_egrid = egrid_file + '_' + str(n) + '_' + str(i)
                    save_path = os.path.join(neg_dir_egrid, save_name_egrid)
                    Save_Document(save_path, Permuted_EGrid, doc_type='egrid') # save negative documents




def Permutation(Document, window, permuted_idx, file_type):
    """
    - np.random.shuffle: Take a list and shuffle the order of the list in place (without return)
    - np.random.permutation: with return

    - window: full indices within a window
    """

    if file_type=='text':
        for i, win in enumerate(window):
            while(win == permuted_idx[i]):
                permuted_idx[i] = list(np.random.permutation(permuted_idx[i])) 

        # Flatten 
        window_flat = np.array(window).flatten()
        permuted_idx_flat = np.array(permuted_idx).flatten()

        Permuted_Document = copy.deepcopy(Document)
        for i in range(len(permuted_idx_flat)):
            Permuted_Document[window_flat[i]] = Document[permuted_idx_flat[i]]
        
        return Permuted_Document, permuted_idx 

    elif file_type=='egrid':
#        pattern = r"^[^- ]*"
#        regex = re.compile(pattern)

        for i, win in enumerate(window): # To prevent the identical permutating indices
            while(win == permuted_idx[i]):
                permuted_idx[i] = list(np.random.permutation(permuted_idx[i])) 

        # Flatten 
        window_flat = np.array(window).flatten()+1 # To consider the space > double the positions
        permuted_idx_flat = np.array(permuted_idx).flatten()+1


        Permuted_Document = copy.deepcopy(Document)
        for j, zipped in enumerate(zip(Document, Permuted_Document)):

            doc, permuted = zipped
            permuted = np.asarray(permuted.split(" "))
            doc = np.asarray(doc.split(" ")) 

            permuted[window_flat] = doc[permuted_idx_flat]
#            for i, perm_idx in enumerate(permuted_idx_flat):
#                permuted[window_flat[i]] = doc[perm_idx]
            permuted = " ".join(list(permuted))
            Permuted_Document[j] = permuted
        
        return Permuted_Document, permuted_idx 
    else:
        print("File Type should be text or egrid")

def Sentence_Window(Document, n_window, window_size, perm, overlap=False):
    """
    - Return possible window selection list
    - Overlap option probably give you some overlaps between windows
    - If you want to make overlaps won't happen, please set overlap as False

    - all permutation: check permutation in itertool library

    n_window: int or 'all'
    window_list: window starting points
    """

    docu_length = len(Document)
    

    assert n_window <= docu_length/window_size, "The number of windows should be smaller"

    window_start = Get_Window_Starting_Points(docu_length, window_size)


    if overlap == False:
        window_idx = Window_Idx(n_window, window_size, window_start, perm) # Return sentence indices within a window
        return window_idx
    else:
        window_idx = Window_Idx(n_window, window_size, window_start, perm)
        return window_idx

def Get_Window_Starting_Points(docu_length, window_size):
    # return a list of window starting points
    # ex) [0,3,6,9]

    window_list  = range(0, docu_length, window_size) 

#    n_start = len(window_list)
#
#    if n_start<21:
#        window_list  = range(0, docu_length, window_size) 
#
#
#    print(f"window_start list: {len(window_list)}") 

    if docu_length-window_list[-1]<window_size: # to handle sentences at the end
        return window_list[:-1]
    else:
        return window_list

def Window_Idx(n_window, window_size, window_start, perm):
    """
    Return sentence indices within a window
    n_window: int or percent
    perm: 
    - all
    - percentage of permutation choices
    """

    if perm==1:
        window = np.random.choice(window_start, n_window, replace=False) # Randomly selects n_windows
        window_idx = []
        for idx in window:
            window_idx.append([i for i in range(idx, idx+window_size)])
        return window_idx

    elif perm=='all':
        """
        Return all possible combinations of window
        """
        window_idx = []
        window_comb = [w_s for w_s in itertools.combinations(window_start, n_window)]
        for comb in window_comb:
            window_temp = []
            for idx in comb:
                window_temp.append([i for i in range(idx, idx+window_size)])
            window_idx.append(window_temp)
        return window_idx

    elif perm > 1:
        """
        Some integer
        window_idx: full indices of windows
        """
        window_idx = []

        window_comb = [w_s for w_s in itertools.combinations(window_start, n_window)] # Generate all possible combinations
        window_comb = Select_N_comb(window_comb, perm)
        
        for comb in window_comb:
            window_temp = []
            for idx in comb:
                window_temp.append([i for i in range(idx, idx+window_size)])
            window_idx.append(window_temp)
        return window_idx

    else:
        print("Errors")

def Select_N_comb(window_comb, N):
    """
    n_selected_windows: number of selected windows
    window_comb: all possible choices of window list
    N: criteria
    """

    n_selected_windows = len(window_comb)

    if n_selected_windows<N:
        n_comb=n_selected_windows
        comb_selection = np.random.choice(range(n_selected_windows), n_comb, replace=False) # Randomly selects some combination indices
        window_comb = [window_comb[comb] for comb in comb_selection] # Selected combinations
        return window_comb

    elif n_selected_windows>=N:
        n_comb=N
        comb_selection = np.random.choice(range(n_selected_windows), n_comb, replace=False) # Randomly selects some combination indices
        window_comb = [window_comb[comb] for comb in comb_selection] # Selected combinations
        return window_comb

    else:
        print("Errors in selecting n_comb")

def Read_Files(path, text=True):
    """
    Search files in a folder within the path
    """

    dir_paths = os.walk(path) # Return Generator
    if text==True:
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

#def Select_EGrid(text_path, egrid_path):
#    Text_List = Read_Files(text_path, text=True)
#    EGrid_List = Read_Files(egrid_path, text=False)
#
#    folder = "./EGrid/"
#    Check_Dir(folder)
#    for text_name in Text_List:
#        pattern = re.escape(str(text_name))+r".+"
#        regex = re.compile(pattern)
#        for egrid_name in EGrid_List:
#            if regex.match(egrid_name):
#                load_path = os.path.join(egrid_path, egrid_name)
#                save_path = os.path.join(folder, egrid_name)
#
#                copyfile(load_path, save_path)


if __name__ =='__main__':
    random.seed(9001)


    #read_path = './test/test/'
    read_path = './raw-wsj/wsj'
    n_window = [1, 2, 3]
    perm = [20, 20, 20] # max # of perm

    dataset = Dataset_Processing(read_path, n_sent=10, n_window=n_window, window_size=3, perm=perm)
    dataset.Create_Dataset()




#        print(f"Permuted_Window_Idx: {permuted_idx}")
#        print("---------------------------------------")
#        print(f"Positive Document: ")
#        Print_by_Lines(Document)
#        print("---------------------------------------")
#        print(f"Negative Document: ")
#        Print_by_Lines(Permuted_Document)
#        sys.exit()

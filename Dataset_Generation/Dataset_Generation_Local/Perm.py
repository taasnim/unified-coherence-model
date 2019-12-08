import sys, os
import numpy as np
import re
import itertools
import argparse
from shutil import copyfile
import json
import copy
import pdb

def Save_Document(save_path, Document, doc_type):
    if doc_type=='text':
        with open(save_path, 'w') as fp:
            for line in Document:
                fp.write(line+'\n')
    else:
        with open(save_path, 'w') as fp:
            for line in Document:
                fp.write(line+'\n')

def Check_Dir(path):
    if not os.path.exists(path): # check directory 
        os.makedirs(path)

def Read_Document_List2(path, file_type):
    """
    1) Read_File_Path
    2) Read_Document
    3) Return training and test documents
    """
    File_Path = Read_File_Path2(path, file_type=file_type)
    print("Path Done")

    neg_document = {}
    pos_document = {}
    egrid_filename_list = []
    neg_name_list = []
    for i, file_path in enumerate(File_Path):
        print(f"i: {i}") 

        file_name = file_path.split('/')[-1]
        egrid_filename_list.append(file_name)

        pos_document[file_name] = Read_Document(file_path, file_type="egrid")

        for i in range(40):
            filename = file_name + "-" + str(i+1)
            neg_name_list.append(filename)
            filepath = file_path + "-" + str(i+1)
            neg_document[filename] = Read_Document(filepath, file_type="egrid")
    print("Negative done") 

#    for file_path in Pos_File_Path:
#
#        file_name = file_path.split('/')[-1]
#        pos_name_list.append(file_name)
#        section = int(file_name.split('.')[0][4:6])
#
#        pos_document[file_name] = Read_Document(file_path, file_type="egrid")


    return neg_document, neg_name_list, pos_document, egrid_filename_list

def Read_Document_List(path, file_type):
    """
    1) Read_File_Path
    2) Read_Document
    3) Return training and test documents
    """
    File_Path, text_name_list, Pos_File_Path = Read_File_Path(path, file_type=file_type)
    print("Path Done")

    document = {}
    pos_document = {}
    egrid_filename_list = []
    pos_name_list = []
    for i, file_path in enumerate(File_Path):
        print(f"i: {i}") 

        file_name = file_path.split('/')[-1]
        egrid_filename_list.append(file_name)
        section = int(file_name.split('.')[0][4:6])

        document[file_name] = Read_Document(file_path, file_type="egrid")
    print("Negative done") 

    for file_path in Pos_File_Path:

        file_name = file_path.split('/')[-1]
        pos_name_list.append(file_name)
        section = int(file_name.split('.')[0][4:6])

        pos_document[file_name] = Read_Document(file_path, file_type="egrid")


#        if section <=13:
#            training_document[file_name]= Read_Document(file_path, file_type)
#        else:
#            test_document[file_name]= Read_Document(file_path, file_type)

    return document, text_name_list, egrid_filename_list, pos_document, pos_name_list

def Read_Document(path, file_type):
    with open(path, 'r', encoding='latin') as f:
        if file_type == 'text':
            #Document = f.readlines()
            Document = [line.strip() for line in  f.readlines()]
            #Document = [line.strip() for line in  Document]
            return np.array(Document)
        else:
            Document = [line.strip() for line in  f.readlines()]
            return np.array(Document)

def Read_File_Path2(path, file_type):
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
                    Text_Path_List.append(Text_Path)

        return Text_Path_List

    elif file_type =='egrid':

        with open("./wsj.test", 'r') as f:
            names = ["./"+"/".join(path.strip().split("/")[-2:]) for path in f.readlines()]

        # Enter subfolders. ex) 01, 02, 03....
        Text_Path_List = []
        for file_path in names:

            filepath = file_path + ".EGrid"
            #Text_Path = os.path.join(file_path)
            Text_Path_List.append(filepath)

        return Text_Path_List
    else:
        print('File type should be a text or egrid')

def Read_File_Path(path, file_type):
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
                    Text_Path_List.append(Text_Path)

        return Text_Path_List

    elif file_type =='egrid':
        pattern = r"^wsj+_+\d+\.+pos+\.+text+\.+parsed+\.+ner+\.+EGrid+-+\d.*$"
        pattern2 = r"^wsj+_+\d+\.+pos+\.+text+\.+parsed+\.+ner+\.+EGrid$"
        regex = re.compile(pattern)
        regex2 = re.compile(pattern2)

        with open("./wsj.test", 'r', encoding='latin') as f:
            names = ["./"+"/".join(path.strip().split("/")[-2:]) for path in f.readlines()]

        # Enter subfolders. ex) 01, 02, 03....
        Text_Path_List = []
        Text_Name_List = []
        Text_Path_Pos_List = []
        for dir_path in dir_paths:
            root_dir = dir_path[0]
            file_path_list = dir_path[2]
#            print(f"root_dir: {root_dir}") 
#            print(f"file_path_list: {file_path_list}") 
#            pdb.set_trace()  

            # Read only text files
            for file_path in file_path_list:
                if regex.match(file_path) is not None:
                    Text_Path = os.path.join(root_dir, file_path)
                    Text_Path_List.append(Text_Path)

                    Text_Name = ".".join(file_path.split(".")[:3])
                    Text_Name_List.append(Text_Name)

                if regex2.match(file_path) is not None:
                    Text_Path = os.path.join(root_dir, file_path)
                    Text_Path_Pos_List.append(Text_Path)

        Text_Name_List = set(Text_Name_List)
        Text_Name_List = list(Text_Name_List)
        return Text_Path_List, Text_Name_List, Text_Path_Pos_List
    else:
        print('File type should be a text or egrid')


if __name__ =='__main__':
    #read_path = './EGrid_train_dev/'
    read_path = './final_test/'
    text_path = './wsj'

    pos_text_folder = './pos_text/'
    neg_text_folder = './neg_text/'

    Check_Dir(pos_text_folder)
    Check_Dir(neg_text_folder)

    train_neg, neg_name_list, train_pos, pos_name_list = Read_Document_List2(read_path, file_type='egrid')
#    train_neg, text_name, neg_name_list, train_pos, pos_name_list = Read_Document_List(read_path, file_type='egrid')
    print("Done loading")
    text_path_list = Read_File_Path(text_path, file_type='text')

    for i, pos_name in enumerate(pos_name_list):
        textname = ".".join(pos_name.split(".")[:3])
        print(f"Textname: {textname}") 

        for textpath in text_path_list:
            textpath_name = textpath.split("/")[-1]
            if textpath_name==textname:
                textname_path = textpath
                break

        text = Read_Document(textname_path, file_type='text')
        text = np.asarray(text)


        egrid = train_pos[pos_name]
        pos_egrid = []
        for entity_line in egrid:
            pos_egrid.append(entity_line.split(" ")[1:])
        pos_egrid = np.asarray(pos_egrid)

        for j in range(40):
            neg = j+1
            neg_name = pos_name+'-'+str(neg)

            try:
                n_egrid = train_neg[neg_name]
            except:
                continue
            neg_egrid = []
            for entity_line in n_egrid:
                neg_egrid.append(entity_line.split(" ")[1:])
            neg_egrid = np.asarray(neg_egrid)

            perm_idx = []
            for pos_col in range(pos_egrid.shape[1]):
                pos_sent = pos_egrid[:, pos_col]

                for neg_col in range(pos_egrid.shape[1]):
                    neg_sent = neg_egrid[:, neg_col]

                    if (pos_sent==neg_sent).all():
                        perm_idx.append(neg_col)
                        break

            save_path = pos_text_folder + textname
            Save_Document(save_path, text, doc_type='text')

            neg_text = text[perm_idx]
            save_path = neg_text_folder + textname + "_" +str(neg)

            Save_Document(save_path, neg_text, doc_type='text')


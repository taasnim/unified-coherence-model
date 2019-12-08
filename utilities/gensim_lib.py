
def Dictionary_Vocab(dir_path):
    """
    Build dictionary of vocabs 
    """
    dictionary = corpora.Dictionary(ReadTxtFiles(dir_path))
    word2idx = dictionary.token2id

    return dictionary, word2idx 

def Add_Dictionary(dictionary, text):
    dictionary.add_documents(text)
    word2idx = dictionary.token2id

    return dictionary, word2idx 

def BoW(dir_path):
    """
    Build bag of words: (wordidx, word_count)
    """
    dictionary = corpora.Dictionary()
    tokenized_list = ReadTxtFiles(args.train_pos)
    bow = [dictionary.doc2bow(tokens, allow_update=True) for tokens in tokenized_list] 
    return bow

import sys, os
import numpy as np
import json
import pickle
import pdb
import time

import re
import argparse
import datetime
import random
import copy

from utilities import logging_tool, DataLoad_word2vec_global
#from LM_utils import TokenBatcher, SoftmaxLoss_utils, SampledSoftmaxLoss_utils
from model_bilinear_word2vec_global import Local_Coherence_Score, BiAffine, Sentence_Embedding_Model, \
    Save_Model_State, AdaptivePairwiseLoss, AdaptivePairwiseLoss_v3, DynamicConv2, DynamicConv4, Linear_Conv, AdaptivePairwiseLossL2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#from fairseq.modules import LightweightConv1dTBC, DynamicConv1dTBC


def Parse_Args():
    parser = argparse.ArgumentParser() 

    # Experiment Setting
    parser.add_argument('--experiment_folder', type=str, default="./Experiments/Bilinear/", help='Save train paired Data')

    # Resume train using the saved model
    parser.add_argument('--resume_train', type=bool, default=False, help='Load saved model and resume training')

    # Dataset parameter
    parser.add_argument('--n_window', type=int, default=3, help='Number of permutation window')
#    parser.add_argument('--train_path', type=str, default="./whole/train/", help='Save train paired Data')
#    parser.add_argument('--test_path', type=str, default="./whole/test/", help='Save test paired Data')
    parser.add_argument('--train_path', type=str, default="../data-global/train/", help='Train paired Data')
    parser.add_argument('--test_path', type=str, default="../data-global/test/", help='Test paired Data')
    parser.add_argument('--file_train_path', type=str, default="../data-global/wsj.train", help='Test paired Data list')
    parser.add_argument('--file_test_path', type=str, default="../data-global/wsj.test", help='Test paired Data list')

    parser.add_argument('--pre_embedding_path', type=str, 
            default="../GoogleNews-vectors-negative300.bin", help='Pretrained word embedding path')
    # Vocab arguments
    parser.add_argument('--vocab_path', type=str, default="../data-tokenized/Vocab", help='Vocab path') 
    #parser.add_argument('--word2idx_path', type=str, default="./Dataset_4/vocab/word2idx", help='word2idx')
    #parser.add_argument('--idx2word_path', type=str, default="./Dataset_4/vocab/idx2word", help='idx2word')

    # Training Parameter-------------------------------------------------------------
    parser.add_argument('--Epoch', type=int, default=25, help='Number of Epoch ')
    parser.add_argument('--learning_rate_step', type=int, default=5, help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--learning_rate_decay', type=float, default=.1, help='Decrease learning rate for every certain epoch ')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Optimizer learning rate')
    parser.add_argument('--ranking_loss_margin', type=float, default=5, help='ranking loss margin')
    parser.add_argument('--device', type=str, default='cpu', help='CPU? GPU?')

    # Minibatch argument
    parser.add_argument('--batch_size', type=int, default=5, help='Mini batch size')
    parser.add_argument('--batch_size_test', type=int, default=5, help='Mini batch size for test')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle items')
    parser.add_argument('--file_types', type=str, default='json', help='Load file type')
    parser.add_argument('--window_size', type=int, default=3, help='Local window size')
    #parser.add_argument('--seed', type=str, default='json', help='Load file type')

    # Network Parameter
    parser.add_argument('--n_vocabs', type=int, help='Word embedding dim, it should be defined using the vocab list')
    parser.add_argument('--embed_dim', type=int, default=300, help='Word embedding dim')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dim of RNN') # This is the best
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    #parser.add_argument('--dropout', type=float, default=0., help='Dropout ratio of RNN')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout ratio of RNN')
    parser.add_argument('--bidirectional', type=bool, default=True, help='Bi-directional RNN?')
    parser.add_argument('--batch_first', type=bool, default=True, help='Dimension order')

    # parser.add_argument('--num_head', type=int, default=16, help='Number of heads in DyConv')
    # parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size of DyConv')
    # parser.add_argument('--conv_dropout', type=float, default=.0, help='DyConv kernel dropout rate')
    # parser.add_argument('--kernel_padding', type=int, default=3, help='DyConv kernel padding')
    # parser.add_argument('--kernel_softmax', type=bool, default=True, help='DyConv kernel softmax')

    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument('--GoogleEmbedding', type=bool, default=True, help='Google embedding')
    embedding.add_argument('--RandomEmbedding', type=bool, default=False, help='Random embedding')
#    embedding.add_argument('--RandomEmbedding', type=bool, default=True, help='Random embedding')
#    embedding.add_argument('--GoogleEmbedding', type=bool, default=False, help='Google embedding')
    # embedding.add_argument('--ELMo', type=bool, default=False, help='ELMo embedding')
    # parser.add_argument('--ELMo_Size', type=str, default='small', help='Size of ELMo')
    parser.add_argument('--bilinear_dim', type=int, default=32, help='bilinear output dim')
    # parser.add_argument('--lm_loss_weight', type=float, default=1.0, help='Lang Model loss weight to be counted in final loss')
    parser.add_argument('--dataset', type=str, default='data-global', help='Which data-set? Options: data-tokenized, data-full, data-global') 
    parser.add_argument('--eval_task', type=str, default='std', help='2 discrimination tasks: std-> standard, inv->inverse') 
    
    return parser.parse_args()

args = Parse_Args()

print("Bilinear Global LM on ", args.dataset)
random.seed(0)  
torch.manual_seed(6) 

now = datetime.datetime.now()
args.experiment_folder = args.experiment_folder + f"{now.year}_{now.month}_{now.day}/"
print(f"experiment_folder: {args.experiment_folder}") 

#logging_tool.ArgLog(args, args.experiment_folder)

#log_path = logging_tool.Log_Path(args.experiment_folder, args.n_window, train=False)
#performance_logger = logging_tool.Setup_Logger('test_info', log_path)

#oov_logger = logging_tool.Setup_Logger('OOV_info', './oov_info.log')

DataLoad_word2vec_global.Print_Args(args)

vocab = DataLoad_word2vec_global.Load_File(args.vocab_path, types='json')
args.n_vocabs = len(vocab)

# Device Setting
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    current_gpu = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_gpu)
    print(f"Running on GPU: {gpu_name}") 
    print(f"arg.device: {args.device}")
else:
    print("Running on CPU")
    args.device = torch.device('cpu')

#word2idx = DataLoad_word2vec_global.Load_File(args.word2idx_path, types='json')
#idx2word = DataLoad_word2vec_global.Load_File(args.idx2word_path, types='json')

word2idx = {tok:i for i, tok in enumerate(vocab)}
idx2word = {i:tok for i, tok in enumerate(vocab)}

if args.dataset == 'data-global':
    print("Reading Global Discrimination Dataset")
    batch_generator = DataLoad_word2vec_global.Batch_Generator_Global(args.train_path, args.file_train_path, args)
    batch_generator_test = DataLoad_word2vec_global.Batch_Generator_Global(args.test_path, args.file_test_path, args, test=True)
else:
    print("Reading Local Discrimination Dataset")
    batch_generator = DataLoad_word2vec_global.Batch_Generator(args.train_path, args)
    batch_generator_test = DataLoad_word2vec_global.Batch_Generator(args.test_path, args, test=True)
#batcher = TokenBatcher(args)

sent_emb_model = Sentence_Embedding_Model(args, hidden_size=args.hidden_dim, output_size=0, vocab=vocab, word2idx=word2idx, idx2word=idx2word, oov_logger=None,
        num_layers=args.num_layers, drp_rate=args.dropout, pad_idx=word2idx['<pad>'])
sent_emb_model = sent_emb_model.to(args.device)

biaffine_layer = BiAffine(args.hidden_dim*2, args.bilinear_dim)
biaffine_layer = biaffine_layer.to(args.device)


# dynamic_conv_layer = DynamicConv4(args)
# dynamic_conv_layer = dynamic_conv_layer.to(args.device) #Its dim is similar to sentence vector dim

coherence_scorer = Local_Coherence_Score(args.bilinear_dim*2)
coherence_scorer = coherence_scorer.to(args.device)

# s_loss = SoftmaxLoss_utils(num_words=len(batcher._lm_vocab._id_to_word), embedding_dim=args.hidden_dim)
# s_loss = s_loss.to(args.device)


#print(f"\n{sent_emb_model}") 
#print(f"\n{biaffine_layer}") 
#print(f"\n{coherence_scorer}") 

local_model = nn.Sequential(sent_emb_model, biaffine_layer, coherence_scorer)
#logging_tool.NetworkLogger(args.experiment_folder, local_model)

if args.resume_train==True:
    model_state = 'Epoch_0_Mini_Batch_2_MMdd_2_13'
    model_load_path = os.path.join(args.model_save_dir, model_state)
    local_model.load_state_dict(torch.load(model_load_path))

optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_step, gamma=args.learning_rate_decay)

criterion = AdaptivePairwiseLoss(margin=args.ranking_loss_margin, device=args.device)

#For LM
# lm_optimizer = torch.optim.Adam(s_loss.parameters(), lr=args.learning_rate, weight_decay=0.00001)
# scheduler_lm = torch.optim.lr_scheduler.StepLR(lm_optimizer, step_size=args.learning_rate_step, gamma=args.learning_rate_decay)


Best_Result = 0
for epoch in range(args.Epoch):
    start_train = time.perf_counter() # Measure one epoch training time

    scheduler.step()
    # scheduler_lm.step()
    local_model = local_model.train()
    # s_loss = s_loss.train()

    N_Data_train = 0 # N_Data is the number of accumulated documents
    N_TP_train = 0
    for n_mini_batch, (batch, batch_doc_len, data_name) in enumerate(batch_generator):
        """
        Batch the document 
        batch -> a list containing minibatch of documents [default 10 documents]
                 every doc contains tokens of positive and negative docs in a separate list
                 10 X 2 X doc_size
        #labels for every tokens. pos_labels contains all 1, neg_labels contains all 1 except 0 in the mismathc position with neg doc
        #   shape(10, #tokens_in_doc)
        #pos_labels, neg_labels = Data_Load.Label_Creator_Batch(batch, args.device) #??? ques in the def of the func
        #pos_batch is a 3D list of [docs->sentences->tokens], similar neg_batch
        #pos_eos is a 2D list of batch [docs->eos_indices], containing list of index of eos ("</s>"") of each positive doc. shape(10, #sentences)
        """

        optimizer.zero_grad()
        # lm_optimizer.zero_grad()

        #pos_batch, neg_batch, pos_eos, neg_eos = DataLoad_word2vec_global.Unpairing_Pos_Neg_new(batch)
        pos_batch, neg_batch = DataLoad_word2vec_global.Unpairing_Pos_Neg(batch)
        #X_ids_forward, X_ids_backward = batcher.batch_sentences(pos_batch[0]) #list of 1d list of numpy arrays
        # X_ids = [batcher.batch_sentences(pos_batch[i]) for i in range(len(pos_batch))] #2d list of numpy arrays pairs
        # X_ids_forward, X_ids_backward = zip(*X_ids)
        # X_ids_forward = list(X_ids_forward)
        # X_ids_backward = list(X_ids_backward)

        #neg_doc_order = DataLoad_word2vec_global.Label_Creator(pos_batch, neg_batch, device=args.device)

        # pos_score = []
        # neg_score = []

        # docu_batch_idx is the sentence level batch
        docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = DataLoad_word2vec_global.Batch_Preprocessing_new(
                pos_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
        docu_batch_idx = docu_batch_idx.to(args.device)


        sent_emb_model.init_hidden(args.batch_size)
        output, hidden = sent_emb_model(docu_batch_idx, modified_batch_sentences_len)
        
        for sign in range(2):
            if sign == 0:
                # docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = \
                #     DataLoad_word2vec_global.Batch_Preprocessing_new(pos_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
                # docu_batch_idx = docu_batch_idx.to(args.device)

                # sent_emb_model.init_hidden(args.batch_size)
                # output, hidden_out = sent_emb_model(docu_batch_idx, modified_batch_sentences_len)
                hidden_out = hidden  
                # output_forward, output_backward = output[:, :, :args.hidden_dim], output[:, :, args.hidden_dim:]
                # lm_loss_forward = torch.mean(torch.stack([s_loss((output_forward[i])[:len(X_ids_forward[i])], torch.from_numpy(X_ids_forward[i]).to(args.device)) for i in range(output_forward.size(0))])) * args.lm_loss_weight
                # lm_loss_backward = torch.mean(torch.stack([s_loss((output_backward[i])[:len(X_ids_backward[i])], torch.from_numpy(X_ids_backward[i]).to(args.device)) for i in range(output_backward.size(0))])) * args.lm_loss_weight
                # lm_loss = lm_loss_forward + lm_loss_backward  
                # d0, d1,d2 = output.size() #[batch_size*max_sentence, max
                # output = output.view(args.batch_size,-1, d1, d2)
                # output_forward, output_backward = output[:, :, :, :args.hidden_dim], output[:, :, :, args.hidden_dim:]

                # lm_loss_forward = torch.mean(torch.stack([s_loss(((output_forward[i])[j])[:len((X_ids_forward[i])[j])], 
                #                                 torch.from_numpy((X_ids_forward[i])[j]).to(args.device))  
                #                                 for i in range(len(X_ids_forward))
                #                                     for j in range(len(X_ids_forward[i]))]))* args.lm_loss_weight

                # lm_loss_backward = torch.mean(torch.stack([s_loss(((output_backward[i])[j])[:len((X_ids_backward[i])[j])], 
                #                                 torch.from_numpy((X_ids_backward[i])[j]).to(args.device))  
                #                                 for i in range(len(X_ids_backward))
                #                                     for j in range(len(X_ids_backward[i]))]))* args.lm_loss_weight

                # lm_loss = lm_loss_forward + lm_loss_backward 
            else: 
                # docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = \
                #     DataLoad_word2vec_global.Batch_Preprocessing_new(neg_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
                # docu_batch_idx = docu_batch_idx.to(args.device)

                # sent_emb_model.init_hidden(args.batch_size)
                # output, hidden_out = sent_emb_model(docu_batch_idx, modified_batch_sentences_len)
                neg_doc_order = DataLoad_word2vec_global.Order_Creator_Global(pos_batch, neg_batch, batch_docs_len, device=args.device)
                hidden_out = torch.zeros_like(hidden)
                for i in range(args.batch_size):
                    hidden_out[i,:,:] = torch.index_select(hidden[i,:,:], dim=0, index=neg_doc_order[i])
                
            '''
            docu_batch_idx -> padded 3D torch tensor of docu_batch [doc, sent, tokens]. contains the id of tokens 
            batch_docs_len -> 1D list of #sentences in each doc. [doc]
            batch_sentences_len -> 2D list containing #tokens of each sentences in each docs [doc->sent]
            modified_batch_sentences_len -> padded version of batch_sentences_len, 2D list [doc->sent]
            '''
            # Global Feature
            # hidden_out = hidden_out.permute(1, 0, 2).contiguous()
            # conv_feature = dynamic_conv_layer(hidden_out)
            # hidden_out = hidden_out.permute(1, 0, 2).contiguous()

            # Local Feature
            index = list(range(hidden_out.size(1)))
            index = index[1:]
            index.append(index[-1])

            forw_idx = torch.LongTensor(index).to(args.device).requires_grad_(False) 
            forward_inputs = torch.index_select(hidden_out, dim=1, index=forw_idx)
            #forward_inputs[:,-1,:] = 0

            bi_curr_inputs = biaffine_layer(hidden_out, forward_inputs)
            bi_forward_inputs = torch.index_select(bi_curr_inputs, dim=1, index=forw_idx)

            cat_bioutput_feat = torch.cat((bi_curr_inputs, bi_forward_inputs), dim=2)
            # mask_val = torch.mean(cat_bioutput_feat, dim=2).unsqueeze(2)

            # conv_extended = conv_feature.repeat(1, cat_bioutput_feat.size(1), 1)
            # coherence_feature = torch.cat((cat_bioutput_feat, conv_extended), dim=2)

            # scores = coherence_scorer(coherence_feature)

            scores = coherence_scorer(cat_bioutput_feat)

            score_mask = DataLoad_word2vec_global.ScoreMask(scores, batch_docs_len, args.device)
                
            masked_score = scores*score_mask

            # Loss masking, loss averaging
            if sign==0:
                pos_score = masked_score 
                # pos_mask = mask_val
                #print(f"pos_score: {pos_score}")
            else:
                neg_score = masked_score
                # neg_mask = mask_val
                #print(f"neg_score: {neg_score}")

        #label = torch.ones(pos_score.size(0)).to(args.device)
        # sub = pos_mask-neg_mask
        # mask = sub!=0 
        # mask = mask.type(torch.FloatTensor).to(args.device)

        #pos_score = pos_score*mask
        #neg_score = neg_score*mask

        # loss = criterion(pos_score, neg_score, mask) + lm_loss
        #loss = criterion(pos_score, neg_score) + lm_loss
        loss = criterion(pos_score, neg_score)

        loss.backward(retain_graph=True)

        optimizer.step()
        # lm_optimizer.step()
        torch.cuda.empty_cache()


        # Document level socre
        if args.device != 'cpu':
            Pos_Doc_Score = np.asarray([score.sum().data.cpu().numpy() for score in pos_score])
            Neg_Doc_Score = np.asarray([score.sum().data.cpu().numpy() for score in neg_score])
        else:
            Pos_Doc_Score = np.asarray([score.sum().data.numpy() for score in pos_local_score]).sum(axis=0)
            Neg_Doc_Score = np.asarray([score.sum().data.numpy() for score in neg_local_score]).sum(axis=0)

        #Score_Comparison = pos_score>neg_score 
        Score_Comparison = Pos_Doc_Score>Neg_Doc_Score 
        Score_Comparison = Score_Comparison*1 # True->1, False->0
        N_Data_train += len(Score_Comparison)
        N_Correct_train = Score_Comparison.sum() # How many documents in a mini-batch are correctly classified
        N_TP_train += N_Correct_train
        if (n_mini_batch+1)%100 == 0:
                print(f"Time: {datetime.datetime.now().time()} || Epoch: {epoch} || N_Mini_Batch: {n_mini_batch} || Mini_Batch_Acc: {sum(Score_Comparison)/args.batch_size}|| Total Loss: {loss}") 
        
        #print(f"Epoch: {epoch} || N_Mini_Batch: {n_mini_batch} || Mini_Batch_Acc: {sum(Score_Comparison)/args.batch_size}||") 

    acc_epoch = N_TP_train / N_Data_train # Accuracy at a certain Epoch
    end_train = time.perf_counter()
    print(f"Training Epoch: {epoch}|| Train accuracy result: {acc_epoch}|| Elapsed Time: {(end_train-start_train)}") 

    print(f"Test set evaluation start...") 
    with torch.no_grad():
        local_model = local_model.eval()
        N_Data = 0 # N_Data is the number of accumulated documents
        N_TP = 0
        for n_mini_batch, (batch, batch_doc_len, data_name) in enumerate(batch_generator_test):
            """
            Batch the document 
            batch -> a list containing minibatch of documents [default 10 documents]
                     every doc contains tokens of positive and negative docs in a separate list
                     10 X 2 X doc_size
            #labels for every tokens. pos_labels contains all 1, neg_labels contains all 1 except 0 in the mismathc position with neg doc
            #   shape(10, #tokens_in_doc)
            #pos_labels, neg_labels = Data_Load.Label_Creator_Batch(batch, args.device) #??? ques in the def of the func
            #pos_batch is a 3D list of [docs->sentences->tokens], similar neg_batch
            #pos_eos is a 2D list of batch [docs->eos_indices], containing list of index of eos ("</s>"") of each positive doc. shape(10, #sentences)
            """

            #pos_batch, neg_batch, pos_eos, neg_eos = DataLoad_word2vec_global.Unpairing_Pos_Neg_new(batch)
            pos_batch, neg_batch = DataLoad_word2vec_global.Unpairing_Pos_Neg(batch)

            pos_score = []
            neg_score = []

            # docu_batch_idx is the sentence level batch
            docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = DataLoad_word2vec_global.Batch_Preprocessing_new(
                    pos_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
            docu_batch_idx = docu_batch_idx.to(args.device)

            sent_emb_model.init_hidden(args.batch_size)
            output, hidden = sent_emb_model(docu_batch_idx, modified_batch_sentences_len)

            for sign in range(2):
                
                if sign == 0:
                    # docu_batch_idx is the sentence level batch
                    # docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = DataLoad_word2vec_global.Batch_Preprocessing_new(
                    #         pos_batch, word2idx, args.batch_size, batch_first=True, pad_token="<pad>")
                    # docu_batch_idx = docu_batch_idx.to(args.device)

                    # sent_emb_model.init_hidden(args.batch_size)
                    # output, hidden = sent_emb_model(docu_batch_idx, modified_batch_sentences_len)
                    hidden_out = hidden  
                else: 
                    if args.eval_task == 'inv':
                        neg_doc_order = DataLoad_word2vec_global.Order_Creator_Inverse(pos_batch, neg_batch, batch_docs_len, device=args.device)
                    else:
                        neg_doc_order = DataLoad_word2vec_global.Order_Creator_Global(pos_batch, neg_batch, batch_docs_len, device=args.device)
                    hidden_out = torch.zeros_like(hidden)
                    for i in range(args.batch_size):
                        hidden_out[i,:,:] = torch.index_select(hidden[i,:,:], dim=0, index=neg_doc_order[i])
                    
                # padding, indexing, sorting
                '''
                docu_batch_idx -> padded 3D torch tensor of docu_batch [doc, sent, tokens]. contains the id of tokens 
                batch_docs_len -> 1D list of #sentences in each doc. [doc]
                batch_sentences_len -> 2D list containing #tokens of each sentences in each docs [doc->sent]
                modified_batch_sentences_len -> padded version of batch_sentences_len, 2D list [doc->sent]
                '''

                # Global Feature
                # hidden_out = hidden_out.permute(1, 0, 2).contiguous()
                # conv_feature = dynamic_conv_layer(hidden_out)
                # hidden_out = hidden_out.permute(1, 0, 2).contiguous()

                # Local Feature
                index = list(range(hidden_out.size(1)))
                index = index[1:]
                index.append(index[-1])

                forw_idx = torch.LongTensor(index).to(args.device).requires_grad_(False) 
                forward_inputs = torch.index_select(hidden_out, dim=1, index=forw_idx)
                #forward_inputs[:,-1,:] = 0

                bi_curr_inputs = biaffine_layer(hidden_out, forward_inputs)
                bi_forward_inputs = torch.index_select(bi_curr_inputs, dim=1, index=forw_idx)

                cat_bioutput_feat = torch.cat((bi_curr_inputs, bi_forward_inputs), dim=2)
                # mask_val = torch.mean(cat_bioutput_feat, dim=2).unsqueeze(2)

                # conv_extended = conv_feature.repeat(1, cat_bioutput_feat.size(1), 1)
                # coherence_feature = torch.cat((cat_bioutput_feat, conv_extended), dim=2)

                # scores = coherence_scorer(coherence_feature)
                scores = coherence_scorer(cat_bioutput_feat)

                score_mask = DataLoad_word2vec_global.ScoreMask(scores, batch_docs_len, args.device)
                    
                masked_score = scores*score_mask

                # Loss masking, loss averaging
                if sign==0:
                    pos_score = masked_score 
                    # pos_mask = mask_val
                    #print(f"pos_score: {pos_score}")
                else:
                    neg_score = masked_score
                    # neg_mask = mask_val
                    #print(f"neg_score: {neg_score}")

                #label = torch.ones(pos_score.size(0)).to(args.device)
                #sub = pos_mask-neg_mask
                #mask = sub!=0 
                #mask = mask.type(torch.FloatTensor).to(args.device)

                #pos_score = pos_score*mask
                #neg_score = neg_score*mask
                
        
            # Document level socre
            if args.device != 'cpu':
                Pos_Doc_Score = np.asarray([score.sum().data.cpu().numpy() for score in pos_score])
                Neg_Doc_Score = np.asarray([score.sum().data.cpu().numpy() for score in neg_score])
            else:
                Pos_Doc_Score = np.asarray([score.sum().data.numpy() for score in pos_local_score]).sum(axis=0)
                Neg_Doc_Score = np.asarray([score.sum().data.numpy() for score in neg_local_score]).sum(axis=0)

            #Score_Comparison = pos_score>neg_score 
            Score_Comparison = Pos_Doc_Score>Neg_Doc_Score 
            Score_Comparison = Score_Comparison*1 # True->1, False->0

            N_Data += len(Score_Comparison)
            N_Correct = Score_Comparison.sum() # How many documents in a mini-batch are correctly classified
            N_TP+=N_Correct

        acc_epoch = N_TP/N_Data # Accuracy at a certain Epoch
        if Best_Result<acc_epoch:
            Best_Result=acc_epoch
            #model_name = f"{args.n_window}_Epoch_{epoch}_MMdd_{now.month}_{now.day}"
            #model_save_path = os.path.join(args.experiment_folder, model_name)
            #Save_Model_State(local_model, model_save_path)

        print(f"Test accuracy result: {acc_epoch}|| Best result so far: {Best_Result}||") 
        #performance_logger.info(f"Epoch: {epoch} || Test accuracy result: {acc_epoch} || Best result so far: {Best_Result}||")


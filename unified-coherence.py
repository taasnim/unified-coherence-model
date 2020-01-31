import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import sys
import os
import time
import datetime
import random

from src import utils, model, lm_model, data_load

parser = utils.argument_parser()
args = parser.parse_args()
if args.ELMo:
    print("**ELMo word Embeddings!")
    parser.set_defaults(learning_rate_step=2,
                        embed_dim=256, GoogleEmbedding=False)
else:
    print("**word2vec Embeddings!")
args = parser.parse_args()


random.seed(0)
torch.manual_seed(6)

now = datetime.datetime.now()
args.experiment_folder = args.experiment_path + \
    f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}/"
if not os.path.exists(args.experiment_folder) and args.save_model:
    os.makedirs(args.experiment_folder)

utils.print_args(args)

# vocabs contain all vocab + <pad>, <bos>, <eos>, <unk>
args.vocabs = utils.load_file(args.vocab_path, file_type='json')
args.n_vocabs = len(args.vocabs)
args.word2idx = {tok: i for i, tok in enumerate(args.vocabs)}
args.idx2word = {i: tok for i, tok in enumerate(args.vocabs)}
args.padding_idx = args.word2idx[args.padding_symbol]

batch_gen_train, batch_gen_test = data_load.create_batch_generators(args)
batcher = lm_model.TokenBatcher(args)
# Sentence encoder
sentence_encoder = model.SentenceEmbeddingModel(args).to(args.device)
# Convolution layer for extracting global coherence patterns
global_feature_extractor = model.LightweightConvolution(args).to(args.device)
# Bilinear layer for modeling inter-sentence relation
bilinear_layer = model.BiAffine(args).to(args.device)
# Linear layer
coherence_scorer = model.LocalCoherenceScore(args).to(args.device)
local_global_model = nn.Sequential(sentence_encoder,
                                   bilinear_layer,
                                   global_feature_extractor,
                                   coherence_scorer)
optimizer = torch.optim.Adam(
    local_global_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=args.learning_rate_step, gamma=args.learning_rate_decay)

# For language model
lm_loss_model = lm_model.SoftmaxLossUtils(num_words=len(
    batcher._lm_vocab.vocabs), embedding_dim=args.hidden_dim).to(args.device)
lm_optimizer = torch.optim.Adam(
    lm_loss_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler_lm = torch.optim.lr_scheduler.StepLR(
    lm_optimizer, step_size=args.learning_rate_step, gamma=args.learning_rate_decay)

criterion = model.AdaptivePairwiseLoss(args)


def calculate_lm_loss(pos_batch, output):
    '''
    pos_batch ->  3d list.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]
    output -> 3D Tensor.  [batch_size X doc_max_len, max_sentence_len, 2*args.hidden_dim] 
    X_ids_forward, X_ids_backward -> 2d list of 1d array. [pdoc0, pdoc1, ..] ->  [arr1, arr2, ..] -> [w_id1, w_id2, ...]
                Needed for Language model
    '''
    X_ids = [batcher.batch_sentences(pos_batch[i])
             for i in range(len(pos_batch))]
    X_ids_forward, X_ids_backward = zip(*X_ids)
    X_ids_forward = list(X_ids_forward)
    X_ids_backward = list(X_ids_backward)

    d0, d1, d2 = output.size()
    output = output.view(args.batch_size_train, -1, d1, d2)
    output_forward, output_backward = output[:, :, :,
                                             :args.hidden_dim], output[:, :, :, args.hidden_dim:]

    lm_loss_forward = torch.mean(torch.stack([lm_loss_model(((output_forward[i])[j])[:len((X_ids_forward[i])[j])],
                                                            torch.from_numpy((X_ids_forward[i])[j]).to(args.device))
                                              for i in range(len(X_ids_forward))
                                              for j in range(len(X_ids_forward[i]))])) * args.lm_loss_weight

    lm_loss_backward = torch.mean(torch.stack([lm_loss_model(((output_backward[i])[j])[:len((X_ids_backward[i])[j])],
                                                             torch.from_numpy((X_ids_backward[i])[j]).to(args.device))
                                               for i in range(len(X_ids_backward))
                                               for j in range(len(X_ids_backward[i]))])) * args.lm_loss_weight
    lm_loss = lm_loss_forward + lm_loss_backward

    return lm_loss


def calculate_scores(batch, test=False):
    '''
    batch -> a 4D list containing minibatch of docs.  [doc0, doc1, ..] -> [pdoc0, ndoc0] -> [sent1, sent2, ..] -> [word1, word2, ..]
    pos_batch/neg_batch -> 3d list.  [pdoc0, pdoc1, ..] -> [sent1, sent2, ..] -> [word1, word2, ..]
    batch_docs_len -> 1D list containing len of docs (num of sentences in each docs)
    batch_sentences_len -> 2D list containing original length of each sentences in each docs [doc->len_sent]
    modified_batch_sentences_len -> 2D numpy array containing len of each sentences in each docs after padding  [doc->len_sent]
    '''
    pos_batch, neg_batch = utils.unpairing_pos_neg(batch)
    if args.ELMo:
        # docu_batch_idx -> 4D Tensor of char_ids for ELMo model [doc->sentences->word->char_ids]
        docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing_elmo(
            pos_batch, args)
    else:
        # docu_batch_idx -> 3D Tensor of word_ids for general embeddings model [doc->sentences->word_ids]
        docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing(
            pos_batch, args)
    '''
    output -> 3D Tensor.  [batch_size X doc_max_len, max_sentence_len, 2*args.hidden_dim] 
    hidden -> 3D Tensor.  [batch_size, doc_max_len, 2*args.hidden_dim] 
    '''
    output, hidden = sentence_encoder(
        docu_batch_idx, modified_batch_sentences_len)

    for doc_type in ['pos', 'neg']:
        if doc_type == 'pos':  # for pos doc
            hidden_out = hidden
            if test == False:  # language model loss calculation only during training
                lm_loss = calculate_lm_loss(pos_batch, output)
        else:
            if test == True and args.eval_task == 'inv':
                neg_doc_order = utils.order_creator_inverse(
                    pos_batch, neg_batch, batch_docs_len, device=args.device)
            else:
                neg_doc_order = utils.order_creator_standard(
                    pos_batch, neg_batch, batch_docs_len, device=args.device)
            hidden_out = torch.zeros_like(hidden)
            for i in range(args.batch_size_train):
                hidden_out[i, :, :] = torch.index_select(
                    hidden[i, :, :], dim=0, index=neg_doc_order[i])

        ### Global Feature ###
        # make the time dim to first, batch to second - for lightweight conv.  [doc_max_len -> batch_size -> 2*args.hidden_dim]
        hidden_out = hidden_out.permute(1, 0, 2).contiguous()
        # 3D Tensor containing global features from lightweight convolution.  [batch -> 1 -> 2*args.hidden_dim]
        # batch is made first dim in the function
        global_features = global_feature_extractor(hidden_out)
        # hidden_out back to original order.  [batch_size -> doc_max_len -> 2*args.hidden_dim]
        hidden_out = hidden_out.permute(1, 0, 2).contiguous()

        ### Local Feature ###

        # Bilinear layer
        # forward_inputs contain 1 index forward to hidden_out, needed in bilinear_layer
        index = list(range(hidden_out.size(1)))
        index = index[1:]
        index.append(index[-1])
        forw_idx = torch.LongTensor(index).to(
            args.device).requires_grad_(False)
        forward_inputs = torch.index_select(
            hidden_out, dim=1, index=forw_idx)
        # 3D Tensor containing output of bilinear layer.   [doc -> sentence -> bilinear_dim]
        bi_curr_inputs = bilinear_layer(hidden_out, forward_inputs)

        # Linear layer
        # bi_forward_inputs contain 1 index forward to bi_curr_inputs, concat them for linear layer which will give local features of consecutive 2 sentences
        bi_forward_inputs = torch.index_select(
            bi_curr_inputs, dim=1, index=forw_idx)
        # 3D Tensor containing local features of consecutive 2 sentences.   [doc -> sentence -> 2*bilinear_dim]
        cat_bioutput_feat = torch.cat(
            (bi_curr_inputs, bi_forward_inputs), dim=2)
        # 3D Tensor containing average values of the local features, needed for calculating loss.   [doc -> sentence -> 1]
        mask_val = torch.mean(cat_bioutput_feat, dim=2).unsqueeze(2)
        # 3D Tensor containing global features repeated by #max_sentence.  [batch -> sentence -> 2*args.hidden_dim]
        conv_extended = global_features.repeat(
            1, cat_bioutput_feat.size(1), 1)
        # 3D Tensor containing concatenated global+local features.  [batch -> sentence -> 2*args.hidden_dim+2*bilinear_dim]
        coherence_feature = torch.cat(
            (cat_bioutput_feat, conv_extended), dim=2)
        # linear layer returns 3D tensor containing scores.   [batch -> sentence -> 1]
        scores = coherence_scorer(coherence_feature)
        # mask value for finding valid scores. valid index contains 1, others 0
        score_mask = utils.score_masked(scores, batch_docs_len, args.device)
        # Only keep the valid scores. 3D tensor containing scores.   [batch -> sentence -> 1]
        masked_score = scores*score_mask

        if doc_type == 'pos':
            pos_score = masked_score
            pos_mask = mask_val
        else:
            neg_score = masked_score
            neg_mask = mask_val

    # Document level socre
    # 1D numpy array containing the sum scores of the document.   [batch]
    pos_doc_score = np.asarray([score.sum().data.cpu().numpy()
                                for score in pos_score])
    neg_doc_score = np.asarray([score.sum().data.cpu().numpy()
                                for score in neg_score])
    score_comparison = pos_doc_score > neg_doc_score
    score_comparison = score_comparison*1  # True->1, False->0

    if test == False:
        # In loss calculation, we don't want to penalize local coherent segments(3sentenes in our case).
        # mask contains local coherent seg info
        sub = pos_mask-neg_mask
        mask = sub != 0
        mask = mask.type(torch.FloatTensor).to(args.device)
        loss = criterion(pos_score, neg_score, mask)
        return loss, lm_loss, score_comparison
    else:
        return score_comparison


Best_Result = 0
for epoch in range(args.Epoch):
    start_train = time.perf_counter()  # Measure one epoch training time
    scheduler.step()
    scheduler_lm.step()
    local_global_model.train()
    lm_loss_model.train()

    n_data_train = 0  # n_data_train is the number of accumulated train documents
    n_TP_train = 0
    for n_mini_batch, (batch, batch_doc_len, data_name) in enumerate(batch_gen_train):
        """
        Batch the document 
        batch -> a 4D list containing minibatch of docs [doc0, doc1, ..] -> [pos, neg] -> [sent1, sent2, ..] -> [word1, word2, ..]
                 every doc contains tokens of positive and negative docs in a separate 2d list
        batch_doc_len -> length (#sentences) of each docs in the batch
        data_name -> name of the pos doc files in the batch
        """
        loss, lm_loss, score_comparison = calculate_scores(batch, test=False)
        total_loss = loss + lm_loss

        optimizer.zero_grad()
        lm_optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lm_optimizer.step()
        torch.cuda.empty_cache()

        n_data_train += len(score_comparison)
        # How many documents in a mini-batch are correctly classified
        n_correct_train = score_comparison.sum()
        n_TP_train += n_correct_train
        if (n_mini_batch+1) % 500 == 0:
            print(f"Time: {datetime.datetime.now().time()} || Epoch: {epoch} || N_Mini_Batch: {n_mini_batch} || Mini_Batch_Acc: {sum(score_comparison)/args.batch_size_train}|| LM loss: {lm_loss}|| Total Loss: {total_loss}")

    acc_epoch = n_TP_train / n_data_train  # Accuracy at a certain Epoch
    end_train = time.perf_counter()
    print(
        f"**Training Epoch: {epoch}|| Train accuracy result: {acc_epoch}|| Elapsed Time: {(end_train-start_train)}")

    print(f"Dev set evaluation start...")
    with torch.no_grad():
        local_global_model.eval()
        lm_loss_model.eval()
        n_data_test = 0  # n_data_test is the number of accumulated test documents
        n_TP_test = 0
        for n_mini_batch, (batch, batch_doc_len, data_name) in enumerate(batch_gen_test):
            score_comparison = calculate_scores(batch, test=True)
            n_data_test += len(score_comparison)
            # How many documents in a mini-batch are correctly classified
            n_correct_test = score_comparison.sum()
            n_TP_test += n_correct_test

        acc_epoch = n_TP_test/n_data_test  # Accuracy at a certain Epoch
        if Best_Result < acc_epoch:
            Best_Result = acc_epoch
            if args.save_model:
                print("***Saving Best Model*****")
                model_name = f"Epoch_{epoch}_MMdd_{now.month}_{now.day}"
                model_save_path = os.path.join(
                    args.experiment_path, model_name)
                torch.save(local_global_model.state_dict(), model_save_path)
        print(
            f"Dev accuracy result: {acc_epoch}|| Best result so far: {Best_Result}||")

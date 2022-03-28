import torch
import torch.nn as nn
import torch.nn.functional as F

import gensim
import numpy as np
import copy

from fairseq.modules import LightweightConv1dTBC

from utilities import utils


class SentenceEmbeddingModel(nn.Module):
    def __init__(self, args):
        super(SentenceEmbeddingModel, self).__init__()
        self.n_vocabs = args.n_vocabs
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.padding_idx = args.padding_idx
        self.dropout = args.dropout
        self.batch_size = args.batch_size_train
        self.dropout_layer = nn.Dropout(self.dropout)
        self.vocabs = args.vocabs
        self.word2idx = args.word2idx
        self.idx2word = args.idx2word
        self.device = args.device
        self.oov_logger = None
        self.ELMo = args.ELMo

        # Embedding Layer
        if args.GoogleEmbedding:
            print("Loading Google Embedding...")
            embeddings = PretrainedEmbeddings(args)
            embeddings.load_pretrained_embeddings()
            self.embeddings = embeddings.create_embedding_layer(trainable=True)
        elif args.RandomEmbedding:
            print("Loading Random Embedding...")
            self.embeddings = nn.Embedding(
                self.n_vocabs, self.embed_dim, padding_idx=self.padding_idx)
        elif args.ELMo:
            self.elmo_embedding = utils.get_ELMo_layer(
                args.ELMo_Size).to(self.device)
        else:
            print("Embedding is not defined")

        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, num_layers=self.num_layers,
                            batch_first=True, bidirectional=True, dropout=self.dropout)

    def forward(self, doc_x, doc_sents_length):
        '''
        doc_x: 3D Tensor of word_ids [doc->sentences->word_ids]
        doc_sents_length: 2D numpy array containing len of each sentences in each docs after padding  [doc->len_sent]
        --both are padded--
        '''
        def contextual_embeddings(data, seq_lengths, insertion_perm=None):
            """
            data -> 3D Tensor  [doc*sentences -> word_ids -> embeddings]
            seq_lengths: 1D np array containing length of sentences in doc*sentences
            idx_sort: sorting idx
            idx_original: original sorting idx
            """
            # Sort by length (keep idx)
            seq_lengths, idx_sort = np.sort(
                seq_lengths)[::-1].copy(), np.argsort(-seq_lengths)
            idx_original = np.argsort(idx_sort)  # original sorting idx
            idx_sort = torch.from_numpy(idx_sort).to(
                self.device)                     # sorting idx
            data = data.index_select(dim=0, index=idx_sort)

            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                data, seq_lengths, batch_first=True)
            packed_output, (h_n, c_n) = self.lstm(
                packed_input, self.init_hidden(data.size(0)))
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True)

            # Un-sort by length
            idx_original = torch.from_numpy(idx_original).to(self.device)
            output = output.index_select(dim=0, index=idx_original)
            h_n = h_n.index_select(dim=1, index=idx_original)
            # we need to manually concatenate the final states for both directions
            fwd_h = h_n[0:h_n.size(0):2]
            bwd_h = h_n[1:h_n.size(0):2]
            h = torch.cat([fwd_h, bwd_h], dim=2)
            return output, h
        if self.ELMo:
            elmo_emb = self.elmo_embedding(doc_x)  # returns a dictionary
            doc_data = (elmo_emb['elmo_representations'][0]).to(
                self.device)  # 0: layer idx
        else:
            doc_data = self.embeddings(doc_x)
        # doc_data -> 4D Tensor.  [doc -> sentences -> word -> embeddings]
        d0, d1, d2, d3 = doc_data.shape
        # first dim in doc_data contains all the sentences
        doc_data = doc_data.view(d0*d1, d2, d3)
        doc_sents_length = doc_sents_length.reshape(d0*d1)
        doc_output, doc_hidden = contextual_embeddings(
            doc_data, doc_sents_length)
        doc_hidden = doc_hidden.view(d0, d1, -1)

        return doc_output, doc_hidden

    def init_hidden(self, batch_size):
        """
        (num_dir*num_layer, batch_size, hidden_dim)
        """
        hidden = torch.zeros(self.num_layers*2, batch_size,
                             self.hidden_dim).to(self.device)
        cell = torch.zeros(self.num_layers*2, batch_size,
                           self.hidden_dim).to(self.device)
        return (hidden, cell)


class PretrainedEmbeddings():
    def __init__(self, args):
        self.pre_embedding_path = args.pre_embedding_path
        self.weights_matrix = np.zeros((len(args.vocabs), args.embed_dim))
        self.model = None
        self.embed_dim = args.embed_dim

        self.vocabs = args.vocabs
        self.n_vocabs = args.n_vocabs
        self.word2idx = args.word2idx
        self.idx2word = args.idx2word
        self.n_word = 0
        self.padding_idx = args.padding_idx
        self.oov_logger = None  # oov_logger

    def load_pretrained_embeddings(self):
        """
        create weights_matrix containing pretrained embeddings of vocab entries.
        if word not found in pretrained list, initialize randomly.
        """
        model = gensim.models.KeyedVectors.load_word2vec_format(
            self.pre_embedding_path, binary=True)
        for word in self.vocabs:
            try:
                self.weights_matrix[self.word2idx[word]] = model[word]
                self.n_word += 1
            except KeyError:
                self.weights_matrix[self.word2idx[word]] = np.random.normal(
                    scale=0.6, size=(self.embed_dim, ))

        print(f"Number of words in vocab: {self.n_vocabs}")
        print(f"Found words in pretrained embeddings: {self.n_word}")
        self.weights_matrix = torch.FloatTensor(self.weights_matrix)

    def create_embedding_layer(self, trainable=True):
        emb_layer = nn.Embedding.from_pretrained(self.weights_matrix)
        if trainable == False:
            emb_layer.weight.requires_grad = False
        return emb_layer


class LightweightConvolution(nn.Module):
    def __init__(self, args):
        super(LightweightConvolution, self).__init__()

        self.first_kernel = args.kernel_size
        self.first_padding = args.kernel_padding

        self.conv_layer_1 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=self.first_kernel, padding_l=self.first_padding,
                                                 num_heads=args.num_head, weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.conv_layer_2 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=self.first_kernel, padding_l=self.first_padding,
                                                 num_heads=args.num_head, weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.conv_layer_3 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=self.first_kernel, padding_l=self.first_padding,
                                                 num_heads=int(args.num_head), weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.conv_layer_4 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=args.kernel_size, padding_l=args.kernel_padding,
                                                 num_heads=int(args.num_head), weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.conv_layer_5 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=args.kernel_size, padding_l=args.kernel_padding,
                                                 num_heads=int(args.num_head), weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.conv_layer_6 = LightweightConv1dTBC(input_size=args.hidden_dim*2, kernel_size=args.kernel_size, padding_l=args.kernel_padding,
                                                 num_heads=int(args.num_head), weight_dropout=args.conv_dropout, weight_softmax=args.kernel_softmax)

        self.relu = nn.ReLU()

    def forward(self, inputs):

        residual = inputs
        output = self.conv_layer_1(inputs)  # reduce dimension /2
        output = self.relu(output)
        output = self.conv_layer_2(output)  # reduce dimension /2
        output = self.relu(output+residual)

        residual = output
        output = self.conv_layer_3(output)  # reduce dimension /2
        output = self.relu(output)
        output = self.conv_layer_4(output)  # reduce dimension /2
        output = self.relu(output+residual)

        residual = output
        output = self.conv_layer_5(output)  # reduce dimension /2
        output = self.relu(output)
        output = self.conv_layer_6(output)  # reduce dimension /2
        output = self.relu(output+residual)

        averaged = torch.mean(output, dim=0).unsqueeze(0)
        # make first dim to batch
        averaged = averaged.permute(
            1, 0, 2).contiguous()  # added to global margin

        return averaged


class BiAffine(nn.Module):
    def __init__(self, args):
        super(BiAffine, self).__init__()
        self.input_dim = args.hidden_dim*2   # *2: Concatenated hidden dim
        self.output_dim = args.bilinear_dim
        self.biaffine = nn.Bilinear(
            self.input_dim, self.input_dim, self.output_dim, bias=True)

    def forward(self, inputs, forward_input):
        output = self.biaffine(inputs, forward_input)
        return output


class LocalCoherenceScore(nn.Module):
    def __init__(self, args):
        """
        - input_dim: input feature dim
        """
        super(LocalCoherenceScore, self).__init__()
        if args.global_model:
            self.input_dim = args.bilinear_dim*2 + args.hidden_dim*2
        else:
            self.input_dim = args.bilinear_dim*2
        self.linear_score = nn.Linear(self.input_dim, 1)

    def forward(self, bi_output):
        output = self.linear_score(bi_output)
        return output


class AdaptivePairwiseLoss(nn.Module):
    def __init__(self, args):
        super(AdaptivePairwiseLoss, self).__init__()
        self.margin = args.ranking_loss_margin
        self.device = args.device

    def forward(self, pos, neg, mask):
        subtract = neg*mask - pos*mask
        mask = mask.type(torch.BoolTensor).to(self.device)

        margin_tensor = torch.zeros_like(subtract)
        margin_tensor.masked_fill_(mask, self.margin)

        max_loss = torch.max(
            margin_tensor+subtract, torch.zeros(pos.size()).to(self.device).requires_grad_(False))
        loss = torch.mean(max_loss)
        return loss

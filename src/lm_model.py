from typing import Set, Tuple
import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from src import utils


class Vocabulary(object):
    def __init__(self, args):
        '''
        filename = the vocabulary file. should contain <unk>, <bos>, <eos>; but not <pad>
        '''
        self.vocabs = utils.load_file(
            args.vocab_path, file_type='json')
        if '<pad>' in self.vocabs:
            self.vocabs.remove('<pad>')
        self._word_to_id = {tok: i for i, tok in enumerate(self.vocabs)}
        self._bos = self._word_to_id['<bos>']
        self._eos = self._word_to_id['<eos>']

    def size(self):
        return len(self.vocabs)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self._word_to_id['<unk>']

    def id_to_word(self, cur_id):
        return self.vocabs[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False):
        """
        sentence -> 1d list containing word tokens as entries.
            convert the sentence to an array of ids, with special tokens added (eos or bos).
            if reverse, then the sentence is assumed to be reversed, and
                this method will swap the BOS/EOS tokens appropriately.
        """
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence]
        if reverse:
            # shifting left by 1, because of prev word prediction in backward rnn
            return np.array([self._bos]+word_ids[:-1], dtype=np.int32)
        else:
            # shifting right by 1, because of next word prediction in forward rnn
            return np.array(word_ids[1:]+[self._eos], dtype=np.int32)


class TokenBatcher(object):
    ''' 
    Batch sentences of tokenized text into token id matrices.
    '''

    def __init__(self, args):
        self.args = args
        self._lm_vocab = Vocabulary(args)

    def batch_sentences(self, docs):
        '''
        docs -> 2d list of sentences  [sent1, sent2, ...] -> [word1, word2, ..]
                Each sentence is a list of tokens without <s> or </s>, e.g.
                [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(docs)
        max_length = max(len(sentence) for sentence in docs)
        X_ids_forward = []
        X_ids_backward = []
        for k, sent in enumerate(docs):
            ids_without_mask = self._lm_vocab.encode(sent, reverse=False)
            X_ids_forward.append(ids_without_mask)
            ids_without_mask = self._lm_vocab.encode(sent, reverse=True)
            X_ids_backward.append(ids_without_mask)
        return X_ids_forward, X_ids_backward


class SoftmaxLossUtils(torch.nn.Module):
    """
    Given some embeddings and some targets, applies a linear layer
    to create logits over possible words and then returns the
    negative log likelihood.
    """

    def __init__(self,
                 num_words: int,
                 embedding_dim: int) -> None:
        super().__init__()

        self.softmax_w = torch.nn.Parameter(torch.randn(
            embedding_dim, num_words) / np.sqrt(embedding_dim))
        self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # embeddings is size (n, embedding_dim)
        # targets is (batch_size, ) with the correct class id
        # Does not do any count normalization / divide by batch size
        probs = torch.nn.functional.log_softmax(torch.matmul(
            embeddings, self.softmax_w) + self.softmax_b, dim=-1)

        return torch.nn.functional.nll_loss(probs, targets.long(), reduction="mean")






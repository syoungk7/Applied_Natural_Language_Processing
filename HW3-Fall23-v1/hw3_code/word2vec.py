import numpy as np
import torch

torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn


class Word2Vec(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocabulary_size = 0

    def tokenize(self, data):
        """
        Split all the words in the data into tokens.

        Args:
            data: (N,) list of sentences in the dataset.

        Return:
            tokens: (N, D_i) list of tokenized sentences. The i-th sentence has D_i words after the split.
        """
        raise NotImplementedError


    def create_vocabulary(self, tokenized_data):
        """
        Create a vocabulary for the tokenized data.
        For each unique word in the vocabulary, assign a unique ID to the word. Please sort the vocabulary before assigning index.

        Assign the word to ID mapping to word2idx variable.
        Assign the ID to word mapping to idx2word variable.
        Assign the size of the vocabulary to vocabulary_size variable.

        Args:
            tokenized_data: (N, D) list of split tokens in each sentence.
            
        Return:
            None (The update is done for self.word2idx, self.idx2word and self.vocabulary_size)
        """
        raise NotImplementedError

    def skipgram_embeddings(self, tokenized_data, window_size=2):
        """
        Create a skipgram embeddings by taking context as middle word and predicting
        N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [1]                       [2]
           [1]                       [3]
           [2]                       [1]
           [2]                       [3]
           [2]                       [4]
           [3]                       [1]
           [3]                       [2]
           [3]                       [4]
           [3]                       [5]
           [4]                       [2]
           [4]                       [3]
           [4]                       [5]
           [5]                       [3]
           [5]                       [4]

        source_tokens: [[1], [1], [2], [2], [2], ...]
        target_tokens: [[2], [3], [1], [3], [4], ...]
        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is the middle word in the window.
            target_tokens: List of elements representing IDs of the context words.
        """
        raise NotImplementedError
    

    def cbow_embeddings(self, tokenized_data, window_size=2):
        """
        Create a cbow embeddings by taking context as N=window_size past words and N=window_size future words.

        NOTE : In case the window range is out of the sentence length, create a
        context by feasible window size. For example : The sentence with tokenIds
        as follows will be represented as
        [1, 2, 3, 4, 5] ->
           source_tokens             target_tokens
           [2,3]                     [1]
           [1,3,4]                   [2]
           [1,2,4,5]                 [3]
           [2,3,5]                   [4]
           [3,4]                     [5]
           
        source_tokens: [[2,3], [1,3,4], [1,2,4,5], [2,3,5], [3,4]]
        target_tokens: [[1], [2], [3], [4], [5]]

        Args:
            tokenized_data: (N, D_i) list of split tokens in each sentence.
            window_size: length of the window for creating context. Default is 2.

        Returns:
            source_tokens: List of elements where each element is maximum of N=window_size*2 context word IDs.
            target_tokens: List of elements representing IDs of the middle word in the window.
        """
        raise NotImplementedError


class SkipGram_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize SkipGram_Model with the embedding layer and a linear layer.
        Please define your embedding layer before your linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(SkipGram_Model, self).__init__()
        self.EMBED_DIMENSION = 300 # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1    # please use this to set max_norm in embedding layer

        raise NotImplementedError

        self.embeddings = # initialize embedding layer
        self.linear = # initialize linear layer

    def forward(self, inputs):
        """
        Implement the SkipGram model architecture as described in the notebook.

        Args:
            inputs: (B, 1) Tensor of IDs for each sentence. Where B = batch size

        Returns:
            output: (B, V) Tensor of logits where V = vocab_size.
            
        Hint:
            No need to have a softmax layer here.
        """
        raise NotImplementedError


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size: int):
        """
        Initialize CBOW_Model with the embedding layer and a linear layer.
        Please define your embedding layer before your linear layer.
        
        Reference: 
            embedding - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
            linear layer - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

        Args:
            vocab_size: Size of the vocabulary.
        """
        super(CBOW_Model, self).__init__()
        self.EMBED_DIMENSION = 300  # please use this to set embedding_dim in embedding layer
        self.EMBED_MAX_NORM = 1     # please use this to set max_norm in embedding layer

        raise NotImplementedError

        self.embeddings = # initialize embedding layer
        self.linear = # initialize linear layer

    def forward(self, inputs):
        """
        Implement the CBOW model architecture as described in the notebook.

        Args:
            inputs: (D_i) Tensor of IDs for each sentence.

        Returns:
            output: (1, V) Tensor of logits
            
        Hints:
            The keepdim parameter may be helpful if using torch.mean
            No need to have a softmax layer here. 
        """
        raise NotImplementedError

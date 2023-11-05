
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class OHE_BOW(object):
	def __init__(self):
		'''
		Initialize instance of OneHotEncoder in self.oh for use in fit and transform
		'''
		self.vocab_size = None
		self.oh = OneHotEncoder()
		self.oh.categories_ = []

	def split_text(self, data):
		'''
		Helper function to separate each string into a list of individual words
		Args: data: list of N strings
		Return: data_split: list of N lists of individual words from each string
		'''
		data_split = []
		if len(data) != 1:
			for i in data:
				data_split.append(i.split())
		else:
			for i in data[0]:
				data_split.append(i.split())
		return data_split

	def flatten_list(self, data):
		'''
		Helper function to flatten a list of list of words into a single list
		Args: data: list of N lists of W_i words
		Return: data_flatten: (W,) numpy array of words,
				where W is the sum of the number of W_i words in each of the list of words
		'''
		data_flatten = []
		for i in data:
			if i not in data_flatten:
				data_flatten += i
		return np.array((data_flatten))


	def fit(self, data):
		'''
		Fit the initialized instance of OneHotEncoder to the given data
		Use split_text to separate the given strings into a list of words and
		flatten_list to flatten the list of words in a sentence into a single list of words
		Set self.vocab_size to the number of unique words in the given data corpus
		Args: data: list of N strings
		Return: None
		Hint: You may find numpy's reshape function helpful when fitting the encoder
		'''
		if len(data) == 1:
			self.tmp = sorted(list(set(self.flatten_list(self.split_text(data[0])))))
		else:
			self.tmp = sorted(list(set(self.flatten_list(self.split_text(data)))))

		self.total_words = list([word, idx] for idx, word in enumerate(self.tmp))
		self.oh.fit(self.total_words)
		self.vocab_size = len(self.total_words)

	def onehot(self, words):
		'''
		Helper function to encode a list of words into one hot encoding format
		Args:,words: list of W_i words from a string
		Return: onehotencoded: (W_i, D) numpy array where:
				W_i is the number of words in the current input list i
				D is the vocab size
		Hint: 	.toarray() may be helpful in converting a sparse matrix into a numpy array
				You can use sklearn's built-in OneHotEncoder transform function
		'''
		t_words = dict((word, idx) for idx, word in enumerate(self.tmp))
		word_to_idx = [t_words[word] for word in words]

		onehotencoded = list()
		for idx in word_to_idx:
			total_lst = [0 for _ in range(self.vocab_size)]
			total_lst[idx] = 1
			onehotencoded.append(total_lst)

		onehotencoded = np.array(onehotencoded)
		return onehotencoded

	def transform(self, data):
		'''
		Use the already fitted instance of OneHotEncoder to help you transform the given 
		data into a bag of words representation. You will need to separate each string 
		into a list of words and iterate through each list to transform into a one hot 
		encoding format.
		Use your one hot encoding of each word in a sentence to get the bag of words count
		representation. You may want to look 
		For any empty strings append a (1, D) array of zeros instead.
		Args: data: list of N strings
		Return: bow: (N, D) numpy array
		Hint: Using a try and except block during one hot encoding transform may be helpful
		'''
		bow = []
		for i in data:
			if len(i) != 1 or 0:
				tmppp = self.onehot(i.split(' '))
				tmp = [sum(x) for x in zip(*tmppp)]
			else:
				tmp = [0 for _ in range(self.vocab_size)]
			bow.append(tmp)

		return np.array(bow)

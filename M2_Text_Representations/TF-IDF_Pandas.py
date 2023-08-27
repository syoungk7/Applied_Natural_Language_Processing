#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:52:50 2023

@author: syoungk
@reference: http://www.sefidian.com/2022/07/28/understanding-tf-idf-with-python-example/
"""

import pandas as pd
import numpy as np

# input
corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]

# word set
words_set = set()
 
for doc in  corpus:
    words = doc.split(' ')
    words_set = words_set.union(set(words))
     
print('Number of words in the corpus:',len(words_set))
print('The words in the corpus: \n', words_set)


# term frequency (TF)
n_docs = len(corpus)
n_words_set = len(words_set)

df_tf = pd.DataFrame(np.zeros((n_docs, n_words_set)), columns=list(words_set))

for i in range(n_docs):
    words = corpus[i].split(' ')
    for w in words:
        df_tf[w][i] = df_tf[w][i] + (1 / len(words))
         
print(df_tf)


# inverse document frequency (IDF)
print("IDF of: ")
 
idf = {}
 
for w in words_set:
    k = 0    # number of documents in the corpus that contain this word
     
    for i in range(n_docs):
        if w in corpus[i].split():
            k += 1
             
    idf[w] =  np.log10(n_docs / k)
    print(f'{w:>15}: {idf[w]:>10}' )


# TF-IDF
df_tf_idf = df_tf.copy()
 
for w in words_set:
    for i in range(n_docs):
        df_tf_idf[w][i] = df_tf[w][i] * idf[w]
         
print(df_tf_idf)
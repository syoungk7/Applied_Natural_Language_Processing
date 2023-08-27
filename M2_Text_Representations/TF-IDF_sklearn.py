#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:58:22 2023

@author: syoungk
@reference: http://www.sefidian.com/2022/07/28/understanding-tf-idf-with-python-example/
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# input
corpus = ['data science is one of the most important fields of science',
          'this is one of the best data science courses',
          'data scientists analyze data' ]


tr_idf_model  = TfidfVectorizer()
tf_idf_vector = tr_idf_model.fit_transform(corpus)
print(type(tf_idf_vector), tf_idf_vector.shape)

tf_idf_array = tf_idf_vector.toarray() 
print(tf_idf_array)

words_set = tr_idf_model.get_feature_names_out() 
print(words_set)

df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set) 
print(df_tf_idf)
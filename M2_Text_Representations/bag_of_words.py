#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:45:26 2023

@author: syoungk
"""

from sklearn.feature_extraction.text import CountVectorizer

my_text = ["it was the best of times, it was the worst of times"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(my_text)

print(X.toarray())
print(vectorizer.get_feature_names_out())
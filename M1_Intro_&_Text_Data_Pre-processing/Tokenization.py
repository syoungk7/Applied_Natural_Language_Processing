#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:09:34 2023

@author: syoungk
"""

#import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "This is a simple sentence."

print(word_tokenize(text))
##  ['This', 'is', 'a', 'simple', 'sentence', '.']
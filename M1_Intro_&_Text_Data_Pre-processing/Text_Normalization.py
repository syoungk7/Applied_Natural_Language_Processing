#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:12:33 2023

@author: syoungk
"""

#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

text = "This is a simple sentence"

# convert to lower case
text = text.lower()

# remove stop words
stop_words = list(stopwords.words('english'))
result = ' '.join([i for i in text.split() if not i in stop_words])

print(result)
## simple sentence
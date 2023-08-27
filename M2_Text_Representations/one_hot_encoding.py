#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 22:36:44 2023

@author: syoungk
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

text = "This is a simple sentence"

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(text.split())

# binary encode
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))
#onehot_encoded.toarray()

print(onehot_encoded.toarray())
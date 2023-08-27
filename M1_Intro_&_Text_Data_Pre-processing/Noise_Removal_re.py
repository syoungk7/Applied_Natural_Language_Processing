#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 23:05:25 2023

@author: syoungk
"""

import re
text = "      # This is a '' simple sentence !!!! 1+ \n"


# remove punctuation and special characters
re.sub('^\s+|\W+|[0-9]|\s+$', ' ', text).strip()

print(re.sub('^\s+|\W+|[0-9]|\s+$', ' ', text).strip())
## This is a simple sentence
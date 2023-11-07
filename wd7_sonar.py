# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:11:49 2023

@author: hama
"""

import pandas as pd

df = pd.read_csv('D:\AI middle class\data\sonar3.csv', header=None)
print(df.head(10))

df.describe()
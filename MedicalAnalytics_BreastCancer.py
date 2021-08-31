# -*- coding: utf-8 -*-
"""
Machine Learning on Breast Cancer
Created on Sun Mar 21 21:55:00 2021
@author: BIVentures
"""

#importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 100)

os.chdir('C:/DirectoryPathHere/')

#importing our cancer dataset
df = pd.read_csv('cancer.csv',index_col= None, na_values='?')

#Get X and Y
X = df.iloc[:,1:31].values
Y = df.iloc[:,:31].values

#print top 5 rows
print(df.head())
print('\n')

#Rest of the code withheld for protection of my work

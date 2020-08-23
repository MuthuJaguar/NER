# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 11:50:41 2020

@author: mk909
"""

import numpy as np
import pandas as pd

data = pd.read_csv('Geneset.csv')

data.fillna('nan', inplace=True)

word = data['WORD'].tolist()
tag = data['TAG'].tolist()

sentence = ['sentence #']
j = 1

for i in word:
   if i == 'nan':
        j = j + 1
        #number = 'sentence'+ ' ' + str(j)
        #sentence.append(number)
   if i != 'nan':
        number = 'sentence'+ ' ' + str(j)
        sentence.append(number)
        
import csv

with open('sentence.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sentence)

ar = np.array(sentence)
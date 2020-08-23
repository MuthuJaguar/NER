# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:47:54 2020

@author: mk909
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import glob

file_names=glob.glob("F:/Machine_Learning/Mymodel/papers/*.txt*")

raw_documents=[]
for  file in file_names:
    try:
        with open(file,"r") as f: raw_documents.append(f.read())
    except:
        pass

raw_documents

print("number of documents",len(raw_documents))


def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

total = concatenate_list_data(raw_documents)

print(len(total))


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
  
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(total) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    
print(len(word_tokens))
print(len(filtered_sentence)) 


































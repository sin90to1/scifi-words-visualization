'''
Author: Alex Shi
Date: 2022-01-14 19:40:38
LastEditTime: 2022-01-14 19:51:49
LastEditors: Alex Shi
Description: 
FilePath: /Course Paper/Course Paper/word_len.py
'''
from audioop import avg
import os
import re
import glob
import string
import json
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import collections
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
pkt_word_tokenizer = WordPunctTokenizer()
pkt_sent_tokenizer = PunktSentenceTokenizer()
word_lemmatizer = WordNetLemmatizer()
punctutation = string.punctuation
all_para_len = {}


def cap_avg_len(filename):
    data = open(filename, 'r').readlines()
    low_stop = [word.lower() for word in stop_words]
    all_words = []
    for para in data:
        sentences = pkt_sent_tokenizer.tokenize(re.sub(r'[^\w\s]', '', para.strip().lower()))
        for sentence in sentences:
            word_list = pkt_word_tokenizer.tokenize(sentence)
            for word in word_list:
                if word not in low_stop:
                    all_words.append(word)
                
    all_len = [len(word) for word in all_words]
    avg_len = np.mean(np.array(all_len))
    return avg_len

def cap_avg_sent_len(filename):
    data = open(filename, 'r').readlines()
    all_len = []
    for para in data:
        sentences = pkt_sent_tokenizer.tokenize(para)
        for sentence in sentences:
            all_len.append(len(pkt_word_tokenizer.tokenize(sentence)))
    return np.mean(np.array(all_len))

def count():
    clark_works = glob.glob('./data/clark/*.txt')
    dune_works = glob.glob('./data/dune/dune_*.txt')
    all_works = clark_works+dune_works
    for file in all_works:
        avg_len = cap_avg_sent_len(file)
        print(f'Average sentence length of {os.path.basename(file)} is {avg_len}')

count()
'''
Author: Alex Shi
Date: 2021-11-08 19:32:27
LastEditTime: 2022-01-14 19:36:40
LastEditors: Alex Shi
Description: This supports my own course paper
FilePath: /Course Paper/Course Paper/word_count.py
'''

'''
Steps:
I. Preprocessing
    1. Word Segmentation
    2. Tags, Symbols, Stopwords
    3. Lemmatization
    4. Store sequencial data into pickle or json files
    5. Build terminology library for dune

II. Analysis
    1. Count no.words in two novels (Here just remove the punctuations 
    in the text -> replace them with space)
    2. Count sentence lengths and average sentence lengths
    3. Extract keywords and high-freq words
    4. Using BERT to train word embedding on both texts
    5. Get semantic information of words
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

def count_word_num(filename):
    raw_txt = open(filename, 'r', encoding='utf-8').readlines()
    all_words = []
    for item in raw_txt:
        all_words.extend(pkt_word_tokenizer.tokenize(re.sub(r'[^\w\s]', '', item.strip().lower())))
    word_num = len(all_words)
    # print(all_words)
    return word_num, all_words

def paragraph_len_count(filename):
    all_paras = open(filename, 'r', encoding='utf-8').readlines()
    all_nums = []
    for para in all_paras:
        sentences = pkt_sent_tokenizer.tokenize(re.sub(r'[^\w\s]', '', para.strip().lower()))
        for sentence in sentences:
            word_list = pkt_word_tokenizer.tokenize(sentence)
            all_nums.append(len(word_list))
    all_nums = np.array(all_nums)
    global all_para_len
    all_para_len[os.path.basename(filename)[:-4]] = all_nums
    return round(np.mean(all_nums))

def word_freq_to_dict(word_list):
    print(word_list[:5])
    count_dict = {}
    squeezed_words = []
    for word in word_list:    # with open('./data/counted_data_useless.json', 'w') as json_file:
    #     json.dump(data, fp=json_file, indent=0)
        if word in stop_words or word in punctutation:
            continue
        squeezed_words.append(word_lemmatizer.lemmatize(word))
        
    for word in squeezed_words:
        if word in stop_words or word in punctutation:
            continue
        count_dict[word] = squeezed_words.count(word)
    
    def take_second(item):
        return item[1]
                
    list_by_freq = [(key, count_dict[key]) for key in count_dict]
    list_by_freq.sort(key=take_second, reverse=True)
    # print(list_by_freq)
    return list_by_freq

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

def main():
    clark_works = glob.glob('./data/clark/*.txt')
    dune_works = glob.glob('./data/dune/dune_*.txt')
    data = {}
    for file in clark_works+dune_works:
        avg_len = cap_avg_len(file)
        # data[os.path.basename(file)[:-4]] = {
        #     'Average Word Length': avg_len
        #                                 # 'Number of Words': word_num, 'Average Paragraph Length': para_len,
        #                                 # '50 Words with Highest Frequency': freq_list[:50]
        #                                 }
    # global all_para_len
    # all_para_len = pd.DataFrame.from_dict(all_para_len, orient='index')
    # sns.boxplot(data=all_para_len.transpose(), width=0.3, palette='Blues')
    # plt.title('Paragraph Length Changes')
    # plt.show()

    with open('./data/counted_data_useless.json', 'w') as json_file:
        json.dump(data, fp=json_file, indent=0)
    df = pd.DataFrame(data)
    df.to_csv('./data/counted_data.csv')
    

def count():
    clark_works = glob.glob('./data/clark/*.txt')
    dune_works = glob.glob('./data/dune/dune_*.txt')
    all_works = clark_works+dune_works
    for file in all_works:
        avg_len = cap_avg_len(file)
        print(f'Average word length of {os.path.basename(avg_len)} is {avg_len}')

    
def visualize_base():
    collected_data = pd.DataFrame(json.loads(open('./data/counted_data_useless.json', 'r').read())).transpose().reset_index()
    sns.barplot(x='index', y='Number of Words', data=collected_data, palette='Blues')
    plt.title("Number of Words in All Fictions")
    plt.show()
    sns.barplot(x='index', y='Average Paragraph Length', data=collected_data, palette='Blues')
    plt.title("Average Paragraph Length in All Fictions")
    plt.show()
    
    
def calculate_ratio():
    all_pkl_files = glob.glob('./data/bin/all*.pkl')
    picked_word_range = 6
    counted = {'all_dune':{}, 'all_clark':{}}
    for file in all_pkl_files:
        file_obj = open(file, 'rb')
        data = pickle.load(file_obj)
        all_word_nums = sum(data.values())
        for i in range(1, picked_word_range+1):
            li = 0
            for item in data.items():
                if item[1] == i:
                    li += 1
            counted[os.path.basename(file)[:-4]]['l'+str(i)] = li/all_word_nums
        file_obj.close()
    counted = pd.DataFrame(counted).transpose().reset_index()
    print(counted)
    counted = counted.melt(id_vars=['index'])
    print(counted)
    sns.barplot(x='variable', y='value', data=counted, hue='index', palette='Blues')
    plt.show()
    
def count_tagging():
    files = glob.glob('./data/all/*.txt')
    res = {}
    all_nums = []
    for file in files:
        all_paras = open(file, 'r', encoding='utf-8').readlines()
        all_tags = []
        for para in all_paras:
            sentences = pkt_sent_tokenizer.tokenize(re.sub(r'[^\w\s]', '', para.strip().lower()))
            for sentence in sentences:
                tokened_sentence = nltk.pos_tag(pkt_word_tokenizer.tokenize(sentence))
                for item in tokened_sentence:
                    all_tags.append(item[1])
        res[os.path.basename(file)[:-4]+'_tags'] = collections.Counter(all_tags)
        all_nums.append(len(all_tags))
    all_nums = np.array(all_nums)
    res = pd.DataFrame(res)
    res.to_csv('./data/tags_count.csv')
    
def cap_high_freq_words(filename):
    all_text = open(filename, 'r').readlines()
    word_list = []
    for line in all_text:
        line = line.strip()
        for word in pkt_word_tokenizer.tokenize(line):
            if word not in stop_words:
                word_list.append(word)
    all_freq_dict = collections.Counter(word_list)
    to_store = []
    for item in all_freq_dict.most_common(100):
        word = item[0]
        if nltk.pos_tag([word])[0][1] == 'NN':
            to_store.append(word)
    open('./data/high_freq_'+os.path.basename(filename), 'w').write(' '.join(to_store))
    

    
if __name__=='__main__':
    # cap_high_freq_words('./data/all/all_clark_train.txt')
    # cap_high_freq_words('./data/all/all_dune_train.txt')
    # cap_avg_len('./data/all/all_dune_train.txt')
    count()
    # df = pd.DataFrame({
    # 'Factor': ['Growth', 'Value'],
    # 'Weight': [0.10, 0.20],
    # 'Variance': [0.15, 0.35]
    # })
    # fig, ax1 = plt.subplots(figsize=(10, 10))
    # tidy = df.melt(id_vars='Factor').rename(columns=str.title)
    # print(tidy)
    # sns.barplot(x='Factor', y='Value', hue='Variable', data=tidy, ax=ax1)
    # sns.despine(fig)
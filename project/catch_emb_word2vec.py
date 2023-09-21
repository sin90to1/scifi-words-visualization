'''
Author: Alex Shi
Date: 2021-12-01 15:32:02
LastEditTime: 2021-12-06 20:33:23
LastEditors: Alex Shi
Description: 
FilePath: /Course Paper/catch_emb_word2vec.py
'''
import math
import collections
import os
import re
import gensim
import pickle
from nltk.util import pr
import numpy as np
import nltk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from numpy.lib.function_base import append
from scipy.sparse import data
from sklearn.manifold import TSNE
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    text = re.sub('[^a-zA-Z1-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def prepare_for_word2vec(filename, target_filename):
    raw_text = open(filename, 'r').read()
    with open(target_filename, 'w') as f:
        for sentence in nltk.sent_tokenize(raw_text):
            print(preprocess_text(sentence.lower()), file=f)

def train_word2vec(filename):
    data = gensim.models.word2vec.LineSentence(filename)
    return Word2Vec(data, vector_size=200, window=5, min_count=1)

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16,9))
    colors = cm.rainbow(np)
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

def tsne_plot_2d(label, data, words=[], a=1):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    embeddings = list(data.values())
    x = [x for x,_ in embeddings]
    y = [y for _,y in embeddings]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    anno_x = []
    anno_y = []
    for word in words:
        anno_x.append(embeddings[list(data.keys()).index(word)][0])
        anno_y.append(embeddings[list(data.keys()).index(word)][1])

    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.8, xy=(anno_x[i], anno_y[i]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig("hhh.png", format='png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__=='__main__': 
    lemmatizer = WordNetLemmatizer()
    sent_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = WordPunctTokenizer()
    # model = KeyedVectors.load_word2vec_format('./data/google/GoogleNews-vectors-negative300.bin', binary=True)
    # clark_text = open('./data/all/all_dune_train.txt', 'r').readlines()
    # prepared_clark_text = [word_tokenizer.tokenize(i) for i in clark_text]
    # print('1')
    # clark_model = Word2Vec(size=300, min_count=3, workers=multiprocessing.cpu_count())
    # clark_model.build_vocab(prepared_clark_text)
    # total_examples = clark_model.corpus_count
    # print('2')
    # clark_model.build_vocab([list(model.vocab.keys())], update=True)
    # print('3')
    # clark_model.intersect_word2vec_format('./data/google/GoogleNews-vectors-negative300.bin', binary=True)
    # print('4')
    # clark_model.train(prepared_clark_text, total_examples=total_examples, epochs=5)
    # print('5')
    # clark_model.init_sims(replace=True)
    # clark_model.save('./model/clark.model')
    # all_words = collections.Counter(sum(prepared_clark_text, []))
    # model = Word2Vec.load('./model/dune.model')
    # tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
    # words = []
    # embedding = []
    # for word in all_words.items():
    #     if word[1] >= 3:
    #         words.append(word[0])
    #         embedding.append(model[word[0]])
    # embedding_2d = tsne_ak_2d.fit_transform(embedding)
    # data = zip(words, embedding_2d)
    # pickle.dump(data, open('./data/embeddings/2d_dune.pkl', 'wb'))
    data = dict(pickle.load(open('./data/embeddings/2d_clark.pkl', 'rb')))
    high_freqs = open('./data/high_freq_all_clark_train.txt', 'r').read().split(' ')
    tsne_plot_2d('Clark', data, high_freqs)
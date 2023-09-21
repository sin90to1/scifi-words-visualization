'''
Author: Alex Shi
Date: 2022-01-14 09:09:08
LastEditTime: 2022-01-14 09:45:31
LastEditors: Alex Shi
Description:
FilePath: /Course Paper/Course Paper/cap_LDA.py
'''

from gensim import models, corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

eng_stop_word = [word.lower() for word in stopwords.words('English')]
corpus = [word_tokenize(item.rstrip()) for item in open('./data/all/all_clark_train.txt', 'r').readlines()]
corpus = [word for word in corpus if word not in eng_stop_word]

def LDA_model(data):
    dictionary = corpora.Dictionary(data)
    corpus = dictionary.doc2bow(data)
    lda = models.ldamodel.LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10)
    return lda
    
if __name__=='__main__':
    corpus = sum([word_tokenize(item.strip()) for item in open('./data/all/all_clark_train.txt', 'r').readlines() if item.rstrip()], [])
    corpus = [word for word in corpus if word not in eng_stop_word]
    print(len(corpus))
    lda_model = LDA_model(corpus)
    topic_words = lda_model.print_topics(num_topics=5, num_words=10)
    print(topic_words)
    
    word_list = lda_model.show_topic(0, 5)
    print(word_list)
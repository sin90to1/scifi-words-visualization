'''
Author: Alex Shi
Date: 2021-11-26 23:34:29
LastEditTime: 2021-12-05 10:24:33
LastEditors: Alex Shi
Description: 
FilePath: /Course Paper/catch_embedding.py
'''

import re
from nltk.util import pr
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clark_data = open('./data/all/all_clark.txt', 'r', encoding='utf-8').readlines()
dune_data = open('./data/all/all_dune.txt', 'r', encoding='utf-8').readlines()
sent_tokenizer = PunktSentenceTokenizer()

def preprocess(raw_list):
    processed_list = []
    for item in raw_list:
        tokened = sent_tokenizer.tokenize(item.strip().lower())
        for sentence in tokened:
            processed_list.append(re.sub(r'[^\w\s]', '', sentence))
            processed_list.append('[SEP]')
    return ['CLS']+processed_list
        
clark_utt = preprocess(clark_data)
dune_utt = preprocess(dune_data)


def input_preparation(text, tokenizer):
    marked_text = '[CLS]' + text + '[SEP]'
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_token = tokenizer.convert_tokens_to_ids(tokenized_text)
    segmented_ids = [1]*len(indexed_token)
    
    tokens_tensor = torch.tensor([indexed_token])
    segment_tensors = torch.tensor([segmented_ids])
    
    return tokenized_text, tokens_tensor, segment_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2][1:]
    
    token_embeddings = hidden_states[-1]
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    list_token_embeddings = [token_emb.tolist() for token_emb in token_embeddings]
    return list_token_embeddings

def capture_embeddings(text_list, model, name, word):
    target_word_embeddings = []
    for text in text_list:
        tokenized_text, tokens_tensor, segment_tensors = input_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segment_tensors, model)
        word_index = tokenized_text.index(word)
        word_embedding = list_token_embeddings[word_index]
        target_word_embeddings.append(word_embedding)
    return target_word_embeddings


if __name__ == '__main__':
    clark_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    dune_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print(tokenizer.tokenize('he drought had lasted now for ten million years, '\
    #                    'and the reign of the terrible lizards had long since ended.'\
    #                    'Here on the Equator, in the continent which would one day be known as Africa, '\
    #                    'the battle for existence had reached a new climax of ferocity, and the victor was not yet in sight. '\
    #                    'In this barren and desiccated land, only the small or the swift or the fierce could flourish, or even hope to survive.'))




# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# clark_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# dune_modle = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# 
# def count_max_len(text):
    # max_len = 0
    # for sent in text:
        # length = len(word_tokenize(sent))
        # if max_len < length:
            # max_len = length
    # return max_len
# 
# max_clark = count_max_len(clark_utt)
# max_dune = count_max_len(dune_utt)
# 
# 
# clark_inputs = tokenizer(clark_utt, return_tensors='tf',
                        #  padding='max_length', truncation=True,
                        #  max_length=max_clark)
# dune_inputs = tokenizer(dune_utt, return_tensors='tf',
                        # padding='max_length', truncation=True,
                        # max_length=max_dune)
# 
# clark_train = clark_model(clark_inputs)
# dune_train = dune_modle(dune_inputs)
# 
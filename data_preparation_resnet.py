# coding: utf-8

import json
import os
from collections import Counter
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops import array_ops
import random
import pickle
import editdistance
import soundex 
import jellyfish
import string
import math

TRAIN_DATA_DIR = 'data/train/'
TEST_DATA_DIR = 'data/test/'

test_data = []
for filename in tqdm.tqdm(os.listdir(TEST_DATA_DIR)):
    with open(os.path.join(TEST_DATA_DIR, filename)) as f:
        d = json.load(f)
    test_data += d

data = []
for filename in tqdm.tqdm(os.listdir(TRAIN_DATA_DIR)):
    with open(os.path.join(TRAIN_DATA_DIR, filename)) as f:
        d = json.load(f)
    data += d

val_indexes = np.random.RandomState(42).choice(np.arange(len(data)), replace=False, size=50000)
val_indexes = set(val_indexes)

train_data = []
val_data = []
for i, item in tqdm.tqdm(enumerate(data)):
    if i in val_indexes:
        val_data.append(item)
    else:
        train_data.append(item)

word_counter = Counter()
for item in tqdm.tqdm(train_data):
    word_counter.update([x['value'].lower() for x in item['entitiesFull'] if x['type'] == 'word'])

word_counter_val = Counter()
for item in tqdm.tqdm(val_data):
    word_counter_val.update([x['value'].lower() for x in item['entitiesFull'] if x['type'] == 'word'])

input_word_counter = Counter()
for item in tqdm.tqdm(train_data):
    input_word_counter.update(['__'+x['type']+'__'+x['value'] if x['type'] != 'url' else '__url__'  for x in item['entitiesShortened']])

rare_input_to_type = {item[0]: '__'+item[0].split('__')[1]+'__'  for item in filter(lambda x: x[1] < 15, input_word_counter.items())}

input_vocab = list(set([rare_input_to_type[x] if x in rare_input_to_type else x for x in input_word_counter.keys()]))

inp_item_to_index = {w: i for i, w in enumerate(input_vocab)}

word_list = [x[0] for x in word_counter.items() if x[1] > 25]
word_list = ['__pass__'] + word_list
word_to_index = {w: i for i, w in enumerate(word_list)}

word_list_val = list(word_counter_val.keys())
word_list_val = ['__pass__'] + word_list_val
word_to_index_val = {w: i for i, w in enumerate(word_list_val)}

len_counter_train = Counter()
len_counter_train.update([len(item['entitiesFull']) for item in train_data])

len_counter_val = Counter()
len_counter_val.update([len(item['entitiesFull']) for item in val_data])

train_user_id_counter = Counter()
train_user_id_counter.update([x['user'] for x in train_data])

val_user_id_counter = Counter()
val_user_id_counter.update([x['user'] for x in val_data])

test_user_id_counter = Counter()
test_user_id_counter.update([x['user'] for x in test_data])

users = list(train_user_id_counter.keys())
users_to_index = {x:i for i, x in enumerate(users)}

len_values_train = list(len_counter_train.items())

len_prob_train = np.array([x[1] for x in len_values_train])
len_prob_train = len_prob_train/len_prob_train.sum()

len_values_train = np.array([x[0] for x in len_values_train])

len_data_train = np.array([len(item['entitiesShortened']) for item in train_data])

input_vocab_set = set(input_vocab)

preprocessed_train_data = []
preprocessed_train_data_users = []

for l in len_values_train:
    inp_indexes = np.zeros((len(np.where(len_data_train==l)[0]), l), dtype=np.int32)
    out_indexes = np.zeros((len(np.where(len_data_train==l)[0]), l), dtype=np.int32)
    u = np.zeros(len(np.where(len_data_train==l)[0]), dtype=np.int32)
    for i, j in tqdm.tqdm(enumerate(np.where(len_data_train==l)[0])):
        item = train_data[j]
        inp_indexes[i] = np.array([inp_item_to_index['__'+x['type']+'__'+x['value']]
              if '__'+x['type']+'__'+x['value'] in input_vocab_set else inp_item_to_index['__'+x['type']+'__']
             for x in item['entitiesShortened']])
        
        out_indexes[i] = np.array([word_to_index[x['value'].lower()]
             if x['type'] == 'word' and x['value'].lower() in word_to_index else 0
             for x in item['entitiesFull']])
        u[i] = users_to_index[item['user']]
    preprocessed_train_data.append((inp_indexes, out_indexes))
    preprocessed_train_data_users.append(u)

len_values_val = list(len_counter_val.items())
len_values_val = np.array([x[0] for x in len_values_val])

len_data_val = np.array([len(item['entitiesShortened']) for item in val_data])

preprocessed_val_data = []
preprocessed_val_data_indexes = []
preprocessed_val_data_users = []

for l in len_values_val:
    inp_indexes = np.zeros((len(np.where(len_data_val==l)[0]), l), dtype=np.int32)
    out_indexes = np.zeros((len(np.where(len_data_val==l)[0]), l), dtype=np.int32)
    u = np.zeros(len(np.where(len_data_val==l)[0]), dtype=np.int32)

    for i, j in tqdm.tqdm(enumerate(np.where(len_data_val==l)[0])):
        item = val_data[j]
        inp_indexes[i] = np.array([inp_item_to_index['__'+x['type']+'__'+x['value']]
              if '__'+x['type']+'__'+x['value'] in input_vocab_set else inp_item_to_index['__'+x['type']+'__']
             for x in item['entitiesShortened']])
        
        out_indexes[i] = np.array([word_to_index_val[x['value'].lower()]
             if x['value'].lower() in word_to_index_val and x['type'] == 'word' else 0
             for x in item['entitiesFull']])
        u[i] = users_to_index[item['user']]
    preprocessed_val_data.append((inp_indexes, out_indexes))
    preprocessed_val_data_indexes.append(np.where(len_data_val==l)[0])
    preprocessed_val_data_users.append(u)


with open('gen_data/val_data.hkl', 'wb') as f:
    pickle.dump(val_data, f)
with open('gen_data/input_vocab.hkl', 'wb') as f:
    pickle.dump(input_vocab, f)
with open('gen_data/word_list.hkl', 'wb') as f:
    pickle.dump(word_list, f)
with open('gen_data/word_list_val.hkl', 'wb') as f:
    pickle.dump(word_list_val, f)
with open('gen_data/users.hkl', 'wb') as f:
    pickle.dump(users, f)
with open('gen_data/len_values_train.hkl', 'wb') as f:
    pickle.dump(len_values_train, f)
with open('gen_data/len_prob_train.hkl', 'wb') as f:
    pickle.dump(len_prob_train, f)
with open('gen_data/preprocessed_train_data_users.hkl', 'wb') as f:
    pickle.dump(preprocessed_train_data_users, f)
with open('gen_data/preprocessed_val_data.hkl', 'wb') as f:
    pickle.dump(preprocessed_val_data, f)
with open('gen_data/preprocessed_val_data_indexes.hkl', 'wb') as f:
    pickle.dump(preprocessed_val_data_indexes, f)
with open('gen_data/preprocessed_val_data_users.hkl', 'wb') as f:
    pickle.dump(preprocessed_val_data_users, f)
with open('gen_data/preprocessed_train_data.hkl', 'wb') as f:
    pickle.dump(preprocessed_train_data, f)
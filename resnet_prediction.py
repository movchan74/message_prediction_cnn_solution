import json
import sys
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
import requests
from requests.auth import HTTPBasicAuth

def conv1D(x, inp, out, kernel, stride, name):
    W = tf.get_variable(name+"W", shape=[kernel, inp, out],
           initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.conv1d(x, W, stride, 'SAME')
    x = layers.bias_add(x)
    return x

def residual_block(inputs, dim, name):
    x = tf.nn.relu(inputs)
    x = conv1D(x, dim, dim, 3, 1, name+'.conv1')
    x = tf.nn.relu(x)
    x = conv1D(x, dim, dim, 3, 1, name+'.conv2')
    return inputs + x

if len(sys.argv) != 3:
    print ('usage: resnet_prediction.py input.json output.json')
    exit()

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename) as f:
    test_data = json.load(f)


with open('gen_data/input_vocab.hkl', 'rb') as f:
    input_vocab = pickle.load(f)
with open('gen_data/word_list.hkl', 'rb') as f:
    word_list = pickle.load(f)
with open('gen_data/users.hkl', 'rb') as f:
    users = pickle.load(f)

input_vocab_set = set(input_vocab)
inp_item_to_index = {w: i for i, w in enumerate(input_vocab)}
users_to_index = {x:i for i, x in enumerate(users)}

emb_dim = 64
vocab_size = len(input_vocab)
user_emb_dim = 64

input_seq = tf.placeholder(tf.int32, shape=[None, None])
output_seq = tf.placeholder(tf.int32, shape=[None, None])
input_user = tf.placeholder(tf.int32, shape=[None, 1])

# Embedding: [vocab_size, emb_dim]
init_width = 0.5 / emb_dim
emb_weights = tf.Variable(
    tf.random_uniform(
        [vocab_size, emb_dim], -init_width, init_width),
    name="embed_weights")

embedding = tf.nn.embedding_lookup(emb_weights, input_seq)

init_width = 0.5 / user_emb_dim
users_emb_weights = tf.Variable(
    tf.random_uniform(
        [len(users), user_emb_dim], -init_width, init_width),
    name="users_embed_weights")

users_embedding = tf.nn.embedding_lookup(users_emb_weights, input_user)

net = conv1D(embedding, emb_dim, 128, 3, 1, 'conv1')
net = residual_block(net, 128, 'res1')
net = residual_block(net, 128, 'res2')
net = residual_block(net, 128, 'res3')
net = tf.concat([tf.tile(users_embedding, [1, tf.shape(net)[1], 1]), net], axis=2)
net = conv1D(net, 192, len(word_list), 1, 1, 'conv_final')

pred_max = tf.argmax(net, 2)
out = tf.reshape(net, (-1, len(word_list)))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(output_seq, (-1,)), logits=out)
loss = tf.reduce_mean(loss)

loss_summary_update = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(loss)

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, 'checkpoints/res_cnn_v2_3.817720.ckpt')

def get_prediction(item):
    inp_indexes = np.array([inp_item_to_index['__'+x['type']+'__'+x['value']]
                  if '__'+x['type']+'__'+x['value'] in input_vocab_set else inp_item_to_index['__'+x['type']+'__']
                 for x in item['entitiesShortened']])

    u = users_to_index[item['user']]



    p = sess.run(net, feed_dict={input_seq: inp_indexes.reshape((1, -1)),
                             input_user: np.array(u).reshape((1, -1))})

    pred_seq = []

    for i, x in enumerate(item['entitiesShortened']):
        if x['type'] != 'letter':
            continue
        for k in np.argsort(p[0][i])[::-1]:
            if word_list[k][0] == x['value'].lower():
                break
    #     k = np.argmax(p[0][i])
        pred_seq.append(x['value']+word_list[k][1:])

    return pred_seq


submission = {}
for item in tqdm.tqdm(test_data):
    submission[item['id']] = get_prediction(item)
    
with open(output_filename, 'w') as f:
    json.dump(submission, f)
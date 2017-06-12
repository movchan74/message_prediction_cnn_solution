# coding: utf-8

import json
import os
import sys
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

if len(sys.argv) != 2:
    print ('usage: train_resnet.py num_iter')
    exit()

num_iter = int(sys.argv[1])

with open('gen_data/val_data.hkl', 'rb') as f:
    val_data = pickle.load(f)
with open('gen_data/input_vocab.hkl', 'rb') as f:
    input_vocab = pickle.load(f)
with open('gen_data/word_list.hkl', 'rb') as f:
    word_list = pickle.load(f)
with open('gen_data/word_list_val.hkl', 'rb') as f:
    word_list_val = pickle.load(f)
with open('gen_data/users.hkl', 'rb') as f:
    users = pickle.load(f)
with open('gen_data/len_values_train.hkl', 'rb') as f:
    len_values_train = pickle.load(f)
with open('gen_data/len_prob_train.hkl', 'rb') as f:
    len_prob_train = pickle.load(f)
with open('gen_data/preprocessed_train_data_users.hkl', 'rb') as f:
    preprocessed_train_data_users = pickle.load(f)
with open('gen_data/preprocessed_val_data.hkl', 'rb') as f:
    preprocessed_val_data = pickle.load(f)
with open('gen_data/preprocessed_val_data_indexes.hkl', 'rb') as f:
    preprocessed_val_data_indexes = pickle.load(f)
with open('gen_data/preprocessed_val_data_users.hkl', 'rb') as f:
    preprocessed_val_data_users = pickle.load(f)
with open('gen_data/preprocessed_train_data.hkl', 'rb') as f:
    preprocessed_train_data = pickle.load(f)

input_vocab_set = set(input_vocab)
inp_item_to_index = {w: i for i, w in enumerate(input_vocab)}
users_to_index = {x:i for i, x in enumerate(users)}

batch_size = 64
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

def get_samples_train():
    l = np.random.choice(np.arange(len(len_values_train)), p=len_prob_train)
    inp_indexes = preprocessed_train_data[l][0]
    out_indexes = preprocessed_train_data[l][1]
    u = preprocessed_train_data_users[l]
    sample = np.random.randint(inp_indexes.shape[0], size=batch_size)
    return inp_indexes[sample], out_indexes[sample], u[sample]

batch_size_val = 64

def compare_pred(x, y):
    if x == y:
        return 1.
    else:
        if editdistance.eval(x, y) <= 1 and jellyfish.soundex(x) == jellyfish.soundex(y):
            return 0.5
    return 0.


writer = tf.summary.FileWriter('./logs/resnet_log_1', graph=tf.get_default_graph())
saver = tf.train.Saver()

step = 0

for i in range(num_iter):
    inp_indexes, out_indexes, u = get_samples_train()

    _, l, summary = sess.run([train_op, loss, summary_op], feed_dict={input_seq:inp_indexes,
                            output_seq:out_indexes, input_user: u.reshape(-1, 1)})
    writer.add_summary(summary, step)
    step = step + 1

    if i%30000 == 0 and i != 0:        
        scores = []
        for t, (inp_indexes_val, out_indexes_val) in enumerate(preprocessed_val_data):
            u = preprocessed_val_data_users[t]
            for i in range(math.ceil(inp_indexes_val.shape[0]/batch_size_val)):
                p = sess.run(pred_max, feed_dict={input_seq:inp_indexes_val[i*batch_size_val:(i+1)*batch_size_val], 
                                                  input_user: u[i*batch_size_val:(i+1)*batch_size_val].reshape([-1, 1])})

                for j in range(p.shape[0]):
                    gt = [word_list_val[k] for k in out_indexes_val[i*batch_size_val+j]]
                    gt_indexes = set([k for k, x in enumerate(gt) if x != '__pass__'])
                    pred = [word_list[p[j, k]] for k in range(p.shape[1]) if k in gt_indexes]
                    scores.append(sum([compare_pred(x, y) for x, y in zip([gt[k] for k in gt_indexes], pred)]))
        saver.save(sess, './checkpoints/resnet_v1_%f.ckpt'%np.array(scores).mean())
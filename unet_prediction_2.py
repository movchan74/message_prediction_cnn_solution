# coding: utf-8

# import sys,os,os.path
# os.environ['CUDA_VISIBLE_DEVICES']=''
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
import requests
from requests.auth import HTTPBasicAuth

def conv1D(x, inp, out, kernel, stride, dilation_rate, name, use_bias=True, activation=None, batch_norm=False):
    try:
        with tf.variable_scope("conv"):
            W = tf.get_variable(name+"W", shape=[kernel, inp, out],
                   initializer=tf.contrib.layers.xavier_initializer())
    except:
        with tf.variable_scope("conv", reuse=True):
            W = tf.get_variable(name+"W", shape=[kernel, inp, out],
                   initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.convolution(
                    input=x,
                    filter=W,
                    strides=(stride,),
                    dilation_rate=(dilation_rate,),
                    padding="SAME",
                    data_format="NWC")
    if use_bias:
        x = layers.bias_add(x)
    if batch_norm:
        x = layers.batch_norm(x)
    if activation is not None:
        x = activation(x)
    return x

def residual_block(inputs, dim, dilation_rate, name):
    x = conv1D(inputs, dim, dim, 3, 1, dilation_rate, name+'.conv1', batch_norm=True, activation=tf.nn.relu)
    x = conv1D(x, dim, dim, 3, 1, dilation_rate, name+'.conv2', batch_norm=True)
    return tf.nn.relu(inputs + x)

def upsampling_1d(x, scale):
    original_shape = tf.shape(x)
    x = tf.image.resize_nearest_neighbor(tf.reshape(x, [-1, original_shape[1], 1, original_shape[2]]),
                                [original_shape[1]*scale, 1])
    x = tf.reshape(x, [-1, original_shape[1]*2, original_shape[2]])
    return x

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    x1_shape_int = tuple([i.__int__() for i in x1.get_shape()])
    
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, 0]
    size = [-1, x2_shape[1], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 2)

if len(sys.argv) != 3:
    print ('usage: unet_prediction_2.py input.json output.json')
    exit()

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename) as f:
    test_data = json.load(f)

max_num_user_mention_per_twit = 126
max_num_hashtag_per_twit = 28

with open('gen_data_2/input_vocab.hkl', 'rb') as f:
    input_vocab = pickle.load(f)
with open('gen_data_2/word_list.hkl', 'rb') as f:
    word_list = pickle.load(f)
with open('gen_data_2/users.hkl', 'rb') as f:
    users = pickle.load(f)

with open('gen_data_2/hashtag_list.hkl', 'rb') as f:
    hashtag_list = pickle.load(f)
with open('gen_data_2/user_mention_list.hkl', 'rb') as f:
    user_mention_list = pickle.load(f)

input_vocab_set = set(input_vocab)
inp_item_to_index = {w: i for i, w in enumerate(input_vocab)}
users_to_index = {x:i for i, x in enumerate(users)}
hashtag_to_index = {w: i for i, w in enumerate(hashtag_list)}
user_mention_to_index = {w: i for i, w in enumerate(user_mention_list)}

emb_dim = 64
vocab_size = len(input_vocab)
user_emb_dim = 64

input_seq = tf.placeholder(tf.int32, shape=[None, None])
output_seq = tf.placeholder(tf.int32, shape=[None, None])

hashtag_input = tf.placeholder(tf.int32, shape=[None, max_num_hashtag_per_twit])
user_mention_input = tf.placeholder(tf.int32, shape=[None, max_num_user_mention_per_twit])
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

hashtag_emb_dim = 64

init_width = 0.5 / hashtag_emb_dim
hashtag_emb_weights = tf.Variable(
    tf.random_uniform(
        [len(hashtag_list), hashtag_emb_dim], -init_width, init_width),
    name="hashtag_embed_weights")

hashtag_embedding = tf.nn.embedding_lookup(hashtag_emb_weights, hashtag_input)
hashtag_embedding = tf.reduce_mean(hashtag_embedding, axis=1)
hashtag_embedding = tf.reshape(hashtag_embedding, (-1, 1, hashtag_emb_dim))

user_mention_emb_dim = 64

init_width = 0.5 / user_mention_emb_dim
user_mention_emb_weights = tf.Variable(
    tf.random_uniform(
        [len(user_mention_list), user_mention_emb_dim], -init_width, init_width),
    name="user_mention_embed_weights")

user_mention_embedding = tf.nn.embedding_lookup(user_mention_emb_weights, user_mention_input)
user_mention_embedding = tf.reduce_mean(user_mention_embedding, axis=1)
user_mention_embedding = tf.reshape(user_mention_embedding, (-1, 1, user_mention_emb_dim))


x = embedding
conv1 = conv1D(x, 64, 64, 3, 1, 1, 'conv1_1', batch_norm=False, activation=tf.nn.relu)
conv1 = conv1D(conv1, 64, 64, 3, 1, 1, 'conv1_2', batch_norm=False, activation=tf.nn.relu)
pool1 = tf.nn.pool(conv1, [2,], "MAX", "SAME", strides=[2,])
conv2 = conv1D(pool1, 64, 128, 3, 1, 1, 'conv2_1', batch_norm=False, activation=tf.nn.relu)
conv2 = conv1D(conv2, 128, 128, 3, 1, 1, 'conv2_2', batch_norm=False, activation=tf.nn.relu)
pool2 = tf.nn.pool(conv2, [2,], "MAX", "SAME", strides=[2,])
conv3 = conv1D(pool2, 128, 256, 3, 1, 1, 'conv3_1_', batch_norm=False, activation=tf.nn.relu)
conv3 = conv1D(conv3, 256, 256, 3, 1, 1, 'conv3_2', batch_norm=False, activation=tf.nn.relu)
conv3 = conv1D(conv3, 256, 256, 3, 1, 1, 'conv3_3', batch_norm=False, activation=tf.nn.relu)
conv3 = tf.concat([tf.tile(users_embedding, [1, tf.shape(conv3)[1], 1]), 
                 tf.tile(user_mention_embedding, [1, tf.shape(conv3)[1], 1]),
                 tf.tile(hashtag_embedding, [1, tf.shape(conv3)[1], 1]),
                 conv3], axis=2)
upsample1 = upsampling_1d(conv3, 2)
upsample1 = crop_and_concat(upsample1, conv2)
conv4 = conv1D(upsample1, 384+user_mention_emb_dim+user_emb_dim+hashtag_emb_dim, 128, 3, 1, 1, 'conv4_1', batch_norm=False, activation=tf.nn.relu)
conv4 = conv1D(conv4, 128, 128, 3, 1, 1, 'conv4_2', batch_norm=False, activation=tf.nn.relu)
upsample2 = upsampling_1d(conv4, 2)
upsample2 = crop_and_concat(upsample2, conv1)
conv5 = conv1D(upsample2, 192, 64, 3, 1, 1, 'conv5_1', batch_norm=False, activation=tf.nn.relu)
conv5 = conv1D(conv5, 64, 64, 3, 1, 1, 'conv5_2', batch_norm=False, activation=tf.nn.relu)
net = conv1D(conv5, 64, len(word_list), 1, 1, 1, 'conv_final')

pred_max = tf.argmax(net, 2)
out = tf.reshape(net, (-1, len(word_list)))

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(output_seq, (-1,)), logits=out)
loss = tf.reduce_mean(loss)

loss_summary_update = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9, epsilon=0.01).minimize(loss)


sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


saver = tf.train.Saver(max_to_keep=0)
#saver.restore(sess, './checkpoints/res_cnn_hashtag_unet_v1_2_2573000_3.855930.ckpt')
saver.restore(sess, './checkpoints/res_cnn_hashtag_unet_v1_2_1003301_3.733280.ckpt')



def entity_shortened_preprocess_test(x):
    if x['type'] == 'url':
        return '__url__'
    elif x['type'] == 'userMention':
        return '__userMention__'
    elif x['type'] == 'hashtag':
        return '__hashtag__'
    else:
        if '__'+x['type']+'__'+x['value'] in input_vocab_set:
            return '__'+x['type']+'__'+x['value']
        else:
            return '__'+x['type']+'__'


def get_prediction(item):
    inp_indexes = np.array([inp_item_to_index[entity_shortened_preprocess_test(x)] for x in item['entitiesShortened']])
    
    hashtag_indexes = np.zeros((1, max_num_hashtag_per_twit), dtype=np.int32)
    user_mention_indexes = np.zeros((1, max_num_user_mention_per_twit), dtype=np.int32)
    hashtag_tmp = np.array([hashtag_to_index[x['value'].lower()] 
         if x['value'].lower() in hashtag_to_index else 0  
         for x in item['entitiesShortened'] if x['type'] == 'hashtag'])
    hashtag_indexes[0, :len(hashtag_tmp)] = hashtag_tmp
    user_mention_tmp = np.array([user_mention_to_index[x['value'].lower()] 
         if x['value'].lower() in user_mention_to_index else 0  
         for x in item['entitiesShortened'] if x['type'] == 'userMention'])
    user_mention_indexes[0, :len(user_mention_tmp)] = user_mention_tmp

    u = users_to_index[item['user']]



    p = sess.run(net, feed_dict={input_seq: inp_indexes.reshape((1, -1)),
                             input_user: np.array(u).reshape((1, -1)),
                             hashtag_input:hashtag_indexes, user_mention_input:user_mention_indexes})

    pred_seq = []

    for i, x in enumerate(item['entitiesShortened']):
        count = 0
        if x['type'] != 'letter':
            continue
        for j, k in enumerate(np.argsort(p[0][i])[::-1]):
            if word_list[k][0] == x['value'].lower():
                count += 1
                if count == 1:
                    break
        pred_seq.append(x['value']+word_list[k][1:])

    return pred_seq



submission = {}
for item in tqdm.tqdm(test_data):
    submission[item['id']] = get_prediction(item)

with open(output_filename, 'w') as f:
    json.dump(submission, f)
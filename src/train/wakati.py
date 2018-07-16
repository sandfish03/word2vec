# coding:UTF-8
from __future__ import print_function

import codecs
import MeCab

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy.random as nr
import sys
import h5py
import math

from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.optimizers import RMSprop
from keras.utils import np_utils

class Prediction :
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.input_dim, self.output_dim, input_length=1, embeddings_initializer=uniform(seed=20180716)))
        model.add(Flatten())
        model.add(Dense(self.input_dim, use_bias=False, kernel_initializer=glorot_uniform(seed=20180716)))
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="RMSprop",metrics=['categorical_accuracy'])
        print('#2')
        return model

    def train(self, x_train, t_train, batch_size, epochs, maxlen, emb_param):
        early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=1, verbose=1)
        print('#1', t_train.shape)
        model = self.create_model()
        #model.load_weights(emb_param)    # 埋め込みパラメーターセット。ファイルをロードして学習を再開したいときに有効にする
        print('#3')

        model.fit(x_train, t_train, batch_size=batch_size, epochs=epochs, verbose=1,
        shuffle=True, callbacks=[early_stopping],validation_split=0.0)
        return model

tagger = MeCab.Tagger("-Owakati")

shiftjis_path = 'kaijin_nijumenso.txt'
utf8_path = 'kaijin_nijumenso_utf8.txt'

fin = codecs.open(shiftjis_path, "r",  "shift_jis")
fout = codecs.open(utf8_path, "w",  "utf-8")

lst = []

for row in fin:
    str_data = tagger.parse(row)
    lst = lst + str_data.split()
fin.close()
fout.close()

mat = np.array(lst)
print(mat.shape)

words = sorted(list(set(mat)))
cnt = np.zeros(len(words))

print('total words:', len(words))

word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索

for j in range(0, len(mat)):
    cnt[word_indices[mat[j]]] += 1

# 出現頻度の少ない単語をUNKで置き換え
words_unk = []

for k in range(0, len(words)):
    if cnt[k] <= 3:
        words_unk.append(words[k])
        words[k] = 'UNK'

print('低頻度語数：', len(words_unk))

words = sorted(list(set(words)))
print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))  # 単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))  # インデックスをキーに単語を検索


maxlen = 10

mat_urtext = np.zeros((len(mat),1), dtype=int)

for i in range(0, len(mat)):
    # 出現頻度の低い単語のインデックスをunkに置き換え
    if mat[i] in word_indices :
        mat_urtext[i,0] = word_indices[mat[i]]
    else :
        mat_urtext[i,0] = word_indices['UNK']

print(mat_urtext.shape)

len_seq = len(mat_urtext) - maxlen
data = []
target = []

for i in range(maxlen, len_seq):
    data.append(mat_urtext[i])
    target.extend(mat_urtext[i-maxlen:i])
    target.extend(mat_urtext[i+1:i+1+maxlen])

x_train = np.array(data).reshape(len(data),1)
t_train = np.array(target).reshape(len(data), maxlen*2)

z = list(zip(x_train, t_train))
nr.seed(12345)
nr.shuffle(z)

x_train, t_train = zip(*z)

x_train = np.array(x_train).reshape(len(data),1)
t_train = np.array(t_train).reshape(len(data),maxlen*2)

print(x_train.shape, t_train.shape)

##############
##
## main
##
##############
vec_dim = 100
epochs = 10
batch_size = 200
input_dim = len(words)
output_dim = vec_dim

emb_param = 'param_skip_gram_2_1.hdf5'
prediction = Prediction(input_dim,output_dim)
row = t_train.shape[0]

t_one_hot = np.zeros((row, input_dim), dtype='int8')

for i in range(0, row):
    for j in range(0, maxlen*2):
        t_one_hot[i, t_train[i,j]] = 1

x_train = x_train.reshape(row,1)
model = prediction.train(x_train, t_one_hot, batch_size,epochs,maxlen,emb_param)

model.save_weights(emb_param)


param_lstm = model.get_weights()
param = param_lstm[0]
word0 = '一'
word1 = '１'
word2 = '２'
vec0 = param[word_indices[word0],:]
vec1 = param[word_indices[word1],:]
vec2 = param[word_indices[word2],:]

vec = vec0 - vec1 + vec2
vec_norm = math.sqrt(np.dot(vec, vec))

w_list = [word_indices[word0], word_indices[word1], word_indices[word2]]
dist = -1.0
m = 0
for j in range(0, 5) :
  dist = -1.0
  m = 0
  for i in range(0, len(words)) :
    if i not in w_list :
      dist0 = np.dot(vec, param[i,:])
      dist0 = dist0 / vec_norm / math.sqrt(np.dot(param[i,:], param[i,:]))
      if dist < dist0 :
        dist = dist0
        m = i
  print('第' + str(j+1) + '候補:')
  print('コサイン類似度=', dist, ' ', m, ' ', indices_word[m])
  w_list.append(m)

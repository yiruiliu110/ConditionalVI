import numpy as np
import codecs
from operator import itemgetter
import collections
import random
import math
import itertools
'''
Created by Lihui Lin. School of Data and Computer Science, Sun Yat-sen University.
This module provides a set of functions for raw text processing and data generating.
Convert words to ids, generate data sets and batches for BOW and sequence modeling.
'''


def load_word_id(vocab_path):
    word_list = []
    word_id = {}
    with codecs.open(vocab_path, 'r', 'utf-8') as data_input:
        for line in data_input:
            word_list.append(line.strip().split()[0])  # read words

    # word:id
    for wid in range(len(word_list)):
        word_id[word_list[wid]] = wid + 1

    # id:word
    word_id_dict = {wid: word for word, wid in word_id.items()}
    return word_id_dict


# load pre-trained word embeddings
def load_emb(emb_url):
    embdin = codecs.open(emb_url, 'r', 'utf-8')
    emb_dict = {}
    for line in embdin:
        items = line.split()
        emb_dict[items[0]] = items[1:] # (word:embedding)
        # emb_dict['unk'] = [0] * (len(items) - 1)
    emb_dict['<eos>'] = np.full(50, 1e-8)
    embdin.close()
    return emb_dict


'''
use for generating a data set of word id vectors, for sequence modeling
input: path of .id file
return: a list of docs represented by lists of word id, the duplicated ids are kept
'''
def create_seq_data_set(data_path):
    data = [] # [[id1, id2, ...], ..., [id_k, ...]
    doc_word_count = [] # [word_count1, word_count2, ...], store the length of each doc
    with codecs.open(data_path, 'r', 'utf-8') as data_input:
        for doc in data_input:
            current_doc = [] # store the ids to represent a doc
            word_count = 0 # number of words in the doc
            for wid in doc.strip().split(): # id1, id2, ...
                current_doc.append(wid) # [id1, id2, ...]
                word_count += 1
            if word_count > 0: # ignore empty lines
                data.append(current_doc)
                doc_word_count.append(word_count)
    return data, doc_word_count


'''
use for generating a data set of word freq vectors, for BOW modeling
input: path of .feat file
return: a list of docs represented by matrix of word embeddings, the duplicated ids are kept
'''
def create_bow_data_set(data_path):
    data = [] # [[(id1:freq1), (id2:freq2), ...], ..., [(id_k:freq_k), (id_(k+1):freq_(k+1))], ...]
    doc_word_count = [] # [word_count1, word_count2, ...], store the length of each doc
    with codecs.open(data_path, 'r', 'utf-8') as data_input:
        for doc in data_input:
            current_doc = {} # store the (id:freq) tuples to represent a doc
            word_count = 0 # number of words in the doc
            for item in doc.strip().split(): # id:freq
                word, freq = item.split(':') # id, freq
                current_doc[word] = freq # (id:freq)
                word_count += int(freq)
            if word_count > 0: # ignore empty lines
                data.append(current_doc)
                doc_word_count.append(word_count)
    return data, doc_word_count


'''
use for generating a data set of word embeddings, for sequence modeling
input: path of .id file
return: a list of docs represented by lists of (word:freq) tuples, the duplicated ids are merged
'''
def create_emb_data_set(data_path, emb_dict, word_id_dict):
    data = []  # [[id1, id2, ...], ..., [id_k, ...]
    doc_word_count = []  # [word_count1, word_count2, ...], store the length of each doc
    with codecs.open(data_path, 'r', 'utf-8') as data_input:
        for doc in data_input:
            current_doc = {}  # store the ids to represent a doc
            word_count = 0  # number of words in the doc
            for wid in doc.strip().split():  # id1, id2, ...
                # skips the docs that only contains unknown words and <eos>
                if word_id_dict[int(wid)] not in emb_dict.keys():
                    continue
                else:
                    current_doc[int(wid)] = emb_dict[word_id_dict[int(wid)]]  # [[1, 2, -3, 3, ...]]
                    word_count += 1
            if word_count > 0:  # ignore empty lines
                data.append(current_doc)
                doc_word_count.append(word_count)
    return data, doc_word_count


def create_batches(input_size, batch_size=64, shuffle=True):
    # read doc in ids
    batches = []
    doc_ids =  list(range(input_size))
    if shuffle:
        random.shuffle(doc_ids)
    for n_batch in range(math.floor(input_size/batch_size)):
        start = n_batch * batch_size
        end = start + batch_size
        batches.append(list(doc_ids[start:end]))

    rest = input_size % batch_size
    if rest > 0:
        # batches.append(list(doc_ids[-rest:]) + [-1]*(batch_size-rest))
        batches.append(list(doc_ids[-rest:]))
    return batches


# all vectors share the same length (vocab_size)
def fetch_word_freq_vectors(data_set, doc_word_count, current_batch, vocab_size):
    batch_size = len(current_batch)
    mask = np.zeros(batch_size)
    batch_data = np.zeros((batch_size, vocab_size+1)) # word_id start from 1
    batch_count = []
    # current batch contains a sequence of doc ids
    for i, doc_id in enumerate(current_batch):
        if doc_id != -1:
            for word_id, freq in data_set[doc_id].items():
                batch_data[i, int(word_id)] = freq  # example: [1, 0, 4, ..., 0, 1]
            mask[i] = 1.0
            batch_count.append(doc_word_count[doc_id]) # example: [[3, 1, 2], [1, 3, 4], ...] (assume vocab_size is 3)
    return batch_data, batch_count, mask


# the vectors have various lengths
def fetch_word_id_vectors(data_set, doc_word_count, current_batch):
    batch_size = len(current_batch)
    mask = np.zeros(batch_size)
    seq_length = [doc_word_count[doc] for doc in current_batch]
    padding = np.max(seq_length)
    batch_data = np.zeros(padding)
    batch_count = []
    # current batch contains a sequence of doc ids
    for i, doc_id in enumerate(current_batch):
        if doc_id != -1:
            for j, word_id in enumerate(data_set[doc_id]):
                batch_data[j] = word_id
            mask[i] = 1.0
            batch_count.append(doc_word_count[doc_id]) # example: [[3, 1, 2], [1, 3, 4, 2, 6], ...]
    return batch_data, batch_count, seq_length, mask


def fetch_emb_matrix(data_set, doc_word_count, current_batch, emb_dim):
    batch_size = len(current_batch)
    mask = np.zeros(batch_size)
    seq_length = [len(data_set[doc]) for doc in current_batch]
    #padding = np.max(seq_length)
    padding = 500
    batch_data = np.zeros((batch_size, padding, emb_dim))  # word_id start from 1
    batch_count = []
    # current batch contains a sequence of doc ids
    for i, doc_id in enumerate(current_batch):
        if doc_id != -1:
            for j, word_id in enumerate(data_set[doc_id]):
                batch_data[i, j]= data_set[doc_id][word_id]
            mask[i] = 1.0
            batch_count.append(doc_word_count[doc_id])  # example: [35, 12, ...]
    return batch_data, batch_count, seq_length, mask


def load_labels(label_path):
    label_dict = {}
    with codecs.open(label_path, 'r', 'utf-8') as labels:
        for i, label in enumerate(labels):
            label_dict[i] = label
        labels.close()
    return label_dict


# split the trainable variables by scope
def variable_parser(var_list, prefix):
  ret_list = []
  for var in var_list:
    varname = var.name
    varprefix = varname.split('/')[0]
    if varprefix == prefix:
      ret_list.append(var)
  return ret_list


def create_seq_data_set_new(data):
    data = [] # [[id1, id2, ...], ..., [id_k, ...]
    doc_word_count = [] # [word_count1, word_count2, ...], store the length of each doc
    for doc in chunk:

        doc_word_ids, doc_word_counts = zip(*doc)

        current_doc = list(itertools.chain(*[[doc_word_id] * doc_word_count for (doc_word_id, doc_word_count) in zip(doc_word_ids, doc_word_counts)]))# store the ids to represent a doc
        word_count = len(current_doc) # number of words in the doc

        data.append(current_doc)
        doc_word_count.append(word_count)
    return data, doc_word_count


# —————— for test ——————



if __name__=='__main__':

    # —————— generate data files ——————

    doc2vocab('./data/NIPS/toy.txt', remove_stop_word=True)
    doc2id('./data/NIPS/train.vocab', './data/NIPS/toy.txt',
           './data/NIPS/toy.txt', './data/NIPS/toy.txt', remove_stop_words=True)
    '''
    # —————— generate data and split into batches ——————
    # load pre-trained word embeddings
    emb_dict = loadEmb('./emb/glove.6B.50d.txt')
    # read id sequence directly and compute the word count
    seq_train_data, seq_train_count = create_seq_data_set('./data/Spooky/train.id')
    # same for valid and test set
    # read the (id:freq) tuples directly and compute the word count
    bow_train_data, bow_train_count = create_bow_data_set('./data/Spooky/train.feat')
    # same for valid and test set
    # read id sequence directly and compute the word count
    emb_train_data, emb_train_count = create_emb_data_set('./data/Spooky/train.id', emb_dict, word_id_dict)
    # same for valid and test set
    # split the data sets into batches of size batch_size
    seq_train_batches = create_batches(len(seq_train_data), 64)
    bow_train_batches = create_batches(len(bow_train_data), 64)
    emb_train_batches = create_batches(len(emb_train_data), 64)
    # map the doc ids in the batches to specific vectors (as the input of networks)
    for seq_batch in seq_train_batches:
        seq_batch_data, seq_batch_count, seq_length, seq_mask = fetch_word_id_vectors(seq_train_data,seq_train_count, seq_batch)
        # training process
    for bow_batch in bow_train_batches:
        bow_batch_data, bow_batch_count, bow_mask = fetch_word_freq_vectors(bow_train_data, bow_train_count, bow_batch, 10000)
        # training process
    for emb_batch in emb_train_batches:
        emb_batch_data, emb_batch_count, seq_length, emb_mask = fetch_emb_matrix(emb_train_data, emb_train_count, emb_batch, 50)
        # training process
    # same for valid set and test sets
'''
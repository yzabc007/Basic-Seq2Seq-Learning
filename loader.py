import os
import re
import sys
import pickle
import numpy as np
import unicodedata
from collections import Counter
from subprocess import Popen, PIPE, STDOUT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
import utils

# class MimicLoader:
#     def __init__(self):
#         super(MimicLoader, self).__init__()


def unicode_to_ascii(s):
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # Lowercase, trim, and remove non-letter characters
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[0].startswith(eng_prefixes)


def txt_loader(file_name):
    src_vocab = []
    tgt_vocab = []
    list_data = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            cur_pairs = [normalize_string(s) for s in line.strip().split('\t')]
            if filter_pair(cur_pairs):
                src_vocab += cur_pairs[0].split(' ')
                tgt_vocab += cur_pairs[1].split(' ')
                list_data.append((cur_pairs[0].split(' '), cur_pairs[1].split(' ')))

    return list_data, src_vocab, tgt_vocab


def batch_processing(batch_data, args):

    encode_seq_ids = []
    encode_seq_len = []
    decode_seq_ids = []
    decode_seq_len = []
    decoder_labels = []

    for data in batch_data:
        encode_seq_ids.append(data['encode_seq_ids'])
        encode_seq_len.append(len(data['encode_seq_ids']))
        decode_seq_ids.append(data['decode_seq_ids'])
        decode_seq_len.append(len(data['decode_seq_ids']))
        decoder_labels.append(data['decoder_labels'])

    encode_seq_ids = utils.pad_sequence(encode_seq_ids, padder=config.PAD_WORD)
    encode_seq_len = np.array(encode_seq_len).astype(np.float32)
    decode_seq_ids = utils.pad_sequence(decode_seq_ids, padder=config.PAD_WORD)
    decode_seq_len = np.array(decode_seq_len).astype(np.float32)
    decoder_labels = utils.pad_sequence(decoder_labels, padder=config.PAD_WORD)

    # sorted_idx = np.argsort(word_lengths)[::-1]

    return [encode_seq_ids, encode_seq_len, decode_seq_ids, decode_seq_len], decoder_labels


def make_idx_data(dataset, args):
    data_dicts = []
    for data in dataset:
        # data: a list of words including segmentation
        cur_dict = {}

        encode_word_ids = [args.src_word2id[w] if w in args.src_word2id else config.UNK_WORD for w in data[0]]
        encode_word_ids = [config.BOS_WORD] + encode_word_ids + [config.EOS_WORD]
        cur_dict['encode_seq_ids'] = encode_word_ids

        decode_word_ids = [args.tgt_word2id[w] if w in args.tgt_word2id else config.UNK_WORD for w in data[1]]
        decode_word_ids = [config.BOS_WORD] + decode_word_ids + [config.EOS_WORD]

        cur_dict['decode_seq_ids'] = decode_word_ids[:-1]
        cur_dict['decoder_labels'] = decode_word_ids[1::]
        data_dicts.append(cur_dict)
    return data_dicts


def raw_word_mapping(vocab):
    word_vocab = Counter(vocab)
    word_vocab = [x[0] for x in word_vocab.items() if x[1] > config.MIN_WORD_FREQ]
    word_to_id = {config.PAD_WORD_S: config.PAD_WORD,
                  config.BOS_WORD_S: config.BOS_WORD,
                  config.EOS_WORD_S: config.EOS_WORD,
                  config.UNK_WORD_S: config.UNK_WORD}

    word_to_id.update({x: idx + 4 for idx, x in enumerate(word_vocab)})
    # word_to_id[config.UNK_WORD] = len(word_to_id) + 1
    # word_to_id[config.PAD_WORD] = len(word_to_id) + 1
    # word_to_id[config.BOS_WORD] = len(word_to_id) + 1
    # word_to_id[config.EOS_WORD] = len(word_to_id) + 1
    id_to_word = {v: k for k, v in word_to_id.items()}
    print('Find {0} words.'.format(len(word_to_id)))
    return word_vocab, word_to_id, id_to_word


def load_pretrain_vec(embed_filename, dim=None):
    word_dict = {}
    with open(embed_filename) as f:
        for idx, line in enumerate(f):
            L = line.strip().split()
            word, vec = L[0], L[1::]
            # word = L[0].lower()
            if dim is None and len(vec) > 1:
                dim = len(vec)
            elif len(vec) == 1:
                print('header? ', L)
                continue
            elif dim != len(vec):
                raise RuntimeError('Wrong dimension!')

            word_dict[word] = np.array(vec, dtype=np.float32)
            # assert(len(word_dict[word]) == input_dim)
    return word_dict


def w2v_mapping_pretrain(vocab, embed_filename, word_embed_dim):
    pretrain_dict = load_pretrain_vec(embed_filename, word_embed_dim)
    pretrain_word_vocab = pretrain_dict.keys()

    full_word_vocab = Counter(vocab)
    word_vocab = [x[0] for x in full_word_vocab.items() if x[0] in pretrain_word_vocab or x[1] > 20]
    # word_vocab = [x[0] for x in full_word_vocab.items() if x[1] > 10]
    word_to_id = {config.PAD_WORD_S: config.PAD_WORD,
                  config.BOS_WORD_S: config.BOS_WORD,
                  config.EOS_WORD_S: config.EOS_WORD,
                  config.UNK_WORD_S: config.UNK_WORD}

    word_to_id.update({x: idx + 4 for idx, x in enumerate(word_vocab)})

    # word_to_id = {x: idx + 1 for idx, x in enumerate(word_vocab)}  # skip 0 id
    # word_to_id[config.UNK_WORD] = len(word_to_id) + 1
    # word_to_id[config.PAD_WORD] = len(word_to_id) + 1
    # word_to_id[config.BOS_WORD] = len(word_to_id) + 1
    # word_to_id[config.EOS_WORD] = len(word_to_id) + 1
    id_to_word = {v: k for k, v in word_to_id.items()}

    '''Glorot & Bengio (AISTATS 2010)'''
    initrange = np.sqrt(6.0 / word_embed_dim)
    W = np.random.uniform(-initrange, initrange, (len(word_to_id) + 1, word_embed_dim))
    W[0] = np.zeros(word_embed_dim)
    i = 0
    for cur_word in word_to_id.keys():
        if cur_word in pretrain_dict:
            i += 1
            W[word_to_id[cur_word]] = pretrain_dict[cur_word]

    print('Find {0} words with pretrain ratio: {1}'.format(len(word_to_id), i / float(len(word_to_id))))
    return word_vocab, word_to_id, id_to_word, W




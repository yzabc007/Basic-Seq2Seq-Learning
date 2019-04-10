import os
import re
import sys
import pickle
import tempfile
import numpy as np
import unicodedata
from collections import Counter
from subprocess import Popen, PIPE, STDOUT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# class MimicLoader:
#     def __init__(self):
#         super(MimicLoader, self).__init__()


def cal_bleu_score(logits, labels, stop_idx=None):
    '''
    :param logits: [B * L * V]; B could be 1
    :param labels: [B * L]
    :return:
    '''
    if isinstance(logits, list):
        # input may be a list of tensors
        pad_pred_seqs = []
        pad_gold_seqs = []
        for i in range(len(logits)):
            cur_pred_seqs = list(np.argmax(logits[i], axis=-1))
            pad_pred_seqs += cur_pred_seqs
            pad_gold_seqs += list(labels[i])
    else:
        pad_pred_seqs = list(np.argmax(logits, axis=-1))
        pad_gold_seqs = list(labels)

    pred_seqs = []
    gold_seqs = []
    for i in range(len(pad_gold_seqs)):
        gold_seqs.append(list(pad_gold_seqs[i][pad_gold_seqs[i].nonzero()]))
        # stop prediction when meet <eos>
        if stop_idx in list(pad_pred_seqs[i]):
            pred_seqs.append(list(pad_pred_seqs[i])[:list(pad_pred_seqs[i]).index(stop_idx)])
        else:
            pred_seqs.append(list(pad_pred_seqs[i]))

    return bleu_score_temp(gold_seqs, pred_seqs)


def bleu_score_temp(reference, output):
    tempfile.tempdir = "./tmp"
    if not os.path.exists(tempfile.tempdir):
        os.makedirs(tempfile.tempdir)

    reference_file = tempfile.NamedTemporaryFile()
    output_file = tempfile.NamedTemporaryFile()

    for seq in reference:
        reference_file.write(b' '.join([bytes(x) for x in seq]) + b'\n')

    for seq in output:
        output_file.write(b' '.join([bytes(x) for x in seq]) + b'\n')

    reference_file.seek(0)
    output_file.seek(0)

    cmd = 'perl multi-bleu.perl {0} < {1}'.format(reference_file.name, output_file.name)
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = str(p.stdout.read())
    bleu = float(re.findall(r'(?<=\\nBLEU = )[\d.]+(?=,)', output)[0])

    reference_file.close()
    output_file.close()
    return bleu


def bleu_score(reference, output):
    with open('reference', 'w') as reference_file:
        for seq in reference:
            reference_file.write(' '.join([str(x) for x in seq]) + '\n')

    with open('output', 'w') as output_file:
        for seq in output:
            output_file.write(' '.join([str(x) for x in seq]) + '\n')

    cmd = 'perl multi-bleu.perl reference < output'
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
    output = str(p.stdout.read())
    bleu = float(re.findall(r'(?<=\\nBLEU = )[\d.]+(?=,)', output)[0])

    return bleu


def masked_cross_entropy(logits, labels):
    '''
    :param logits: [(B * L) * V]
    :param labels: [(B * L)]
    :return:
    '''
    labels_flat = labels.reshape(-1)
    logits_flat = logits.reshape(labels_flat.shape[0], -1)
    log_logits = F.log_softmax(logits_flat, dim=1)
    raw_loss = torch.gather(log_logits, dim=1, index=labels_flat.reshape(-1, 1))  # (B*L) * 1
    mask_idx = torch.nonzero(labels_flat)  # 0s in labels mean padded ones
    loss = -torch.sum(torch.gather(raw_loss, dim=0, index=mask_idx))
    # return loss
    return loss / logits.shape[-1]


def sorted_rnn_cell(input_seqs, intput_lens, embeddings, rnn, batch_first=True, last_hidden=None):
    '''rnn cell for sorted sequence'''
    seq_lengths, perm_idx = intput_lens.sort(0, descending=True)
    # print(seq_lengths)
    _, unperm_idx = perm_idx.sort(0)
    input_seqs = input_seqs[perm_idx]
    embed_x = embeddings(input_seqs)  # (B, L, D)
    pack_x = pack_padded_sequence(embed_x, seq_lengths, batch_first=batch_first)
    packed_output, (h_t, c_t) = rnn(pack_x)  # h_t/c_t: 2 * B * D

    lstm_output, a = pad_packed_sequence(packed_output, batch_first=batch_first)  # B * L * 2D
    lstm_output = lstm_output[unperm_idx]

    last_h_t = torch.transpose(h_t, 0, 1)[unperm_idx]
    last_c_t = torch.transpose(c_t, 0, 1)[unperm_idx]
    last_h_t = torch.transpose(last_h_t, 0, 1).contiguous()
    last_c_t = torch.transpose(last_c_t, 0, 1).contiguous()
    hidden = (last_h_t, last_c_t)

    return lstm_output, hidden


def pad_sequence(list_ids, min_length=0, max_length=None, padder=0):
    if not max_length:
        max_length = max([len(x) for x in list_ids] + [min_length])
    # print(max_length)
    new_list_ids = []
    for ids in list_ids:
        new_list_ids.append(ids + [padder] * (max_length - len(ids)))
    # return torch.tensor(new_list_ids)
    return np.array(new_list_ids)


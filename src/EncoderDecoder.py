import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import utils
import loader
import config


class Encoder(nn.Module):
    '''
    Batched RNN Encoder
    '''
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.vocab_size = args.src_vocab_size
        self.word_dim = args.word_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.word_dim)
        self.init_embedding()
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(self.word_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False, batch_first=True)

    def init_embedding(self):
        initrange = 0.5 / self.word_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def initHidden(self):
        return torch.zeros(1, 1, self.lstm_hidden_dim, device=self.args.device)

    def forward(self, input_seqs, intput_lens=None, hidden=None):
        embed_x = self.dropout(self.embeddings(input_seqs))  # (B, L, D)
        lstm_output, hidden = self.lstm(embed_x, hidden)  # h_t/c_t: 2 * B * D

        # lstm_output, hidden = utils.sorted_rnn_cell(input_seqs, intput_lens, self.embeddings, self.lstm, True, hidden)

        return lstm_output, hidden


class Decoder(nn.Module):
    '''
    Sequence to Sequence Learning with Neural Networks
    Batched RNN Decoder
    Notes:
        1. Cannot use bi-directional RNNs
        2. intput_lens is only useful when using mini-batching
    '''
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.vocab_size = args.tgt_vocab_size
        self.word_dim = args.word_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.word_dim)
        self.init_embedding()
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(self.word_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False, batch_first=True)

        self.decoder_out = nn.Linear(self.lstm_hidden_dim, self.vocab_size)

    def init_embedding(self):
        initrange = 0.5 / self.word_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, intput_lens=None, last_hidden=None):
        embed_x = self.dropout(self.embeddings(input_seqs))  # (B, L, D)
        # embed_x = F.relu(embed_x)
        lstm_output, hidden = self.lstm(embed_x, last_hidden)  # B * L * D

        # lstm_output, hidden = utils.sorted_rnn_cell(input_seqs, intput_lens, self.embeddings, self.lstm, True, hidden)

        logits = self.decoder_out(lstm_output)
        # logits = logits.reshape(-1, self.vocab_size)
        return logits, hidden


class ContextDecoder(nn.Module):
    '''
    Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
    '''
    def __init__(self, args):
        super(ContextDecoder, self).__init__()
        self.vocab_size = args.tgt_vocab_size
        self.word_dim = args.word_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.word_dim)
        self.init_embedding()
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(self.word_dim + self.lstm_hidden_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False, batch_first=True)

        self.decoder_out = nn.Linear(self.lstm_hidden_dim * 2, self.vocab_size)

    def init_embedding(self):
        initrange = 0.5 / self.word_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, context, intput_lens=None, last_hidden=None):
        sent_len = input_seqs.shape[1]
        embed_x = self.dropout(self.embeddings(input_seqs))  # [B, L, d1]
        context = torch.transpose(context, 0, 1)  # TODO: only work when num_layers==1
        copy_context = context.repeat(1, sent_len, 1)  # [B, 1, d2] -> [B, L, d2]
        embed_context = torch.cat((embed_x, copy_context), dim=-1)  # [B, L, (d1+d2)]
        lstm_output, hidden = self.lstm(embed_context, last_hidden)  # [B, L, d3]

        lstm_context = torch.cat((lstm_output, copy_context), dim=-1)  # [B, L, (d3+d2)]
        logits = self.decoder_out(lstm_context)
        # logits = logits.reshape(-1, self.vocab_size)
        return logits, hidden


class AttentionModule(nn.Module):
    def __init__(self, args):
        super(AttentionModule, self).__init__()
        self.args = args
        self.att_method = args.att_method
        self.lstm_hidden_dim = args.lstm_hidden_dim

        # define parameters
        if self.att_method == 'general':
            self.att_bi_fc = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim)
        elif self.att_method == 'concat':
            self.att_fc_1 = nn.Linear(self.lstm_hidden_dim * 2, self.lstm_hidden_dim)
            self.att_fc_2 = nn.Linear(self.lstm_hidden_dim, 1)

    def forward(self, query, keys, values=None, mask=None):
        # query: [B, 1, d1]
        # keys: [B, L, d2] - padding?
        if not values:
            values = keys

        batch_size = query.shape[0]
        max_src_len = keys.shape[1]

        # find a way to produce a scalar between two vectors (batched ones)
        if self.att_method == 'dot':
            assert query.shape[-1] == keys.shape[-1], 'Not matched dim!'
            keys = keys.permute(0, 2, 1)  # [B, d2, L]
            energy = torch.bmm(query, keys)  # [B, 1, d1] *  [B, d2, L] -> [B, 1, L]

        elif self.att_method == 'general':
            energy = self.att_bi_fc(query)  # [B, 1, d1']
            keys = keys.permute(0, 2, 1)  # [B, d2, L]
            energy = torch.bmm(energy, keys)  # [B, 1, d1'] *  [B, d2, L] -> [B, 1, L]

        elif self.att_method == 'concat':
            query = query.repeat(1, max_src_len, 1)  # [B, L, d1]
            energy = self.att_fc_1(torch.cat((query, keys), dim=-1))  # [B, L, (d1+d2)] -> [B, L, d]
            energy = self.att_fc_2(torch.tanh(energy))  # [B, L, 1]
            energy = energy.permute(0, 2, 1)  # [B, 1, L]
        else:
            raise ValueError

        energy = energy.masked_fill_(mask.unsqueeze(1), -1e10)
        att_weight = F.softmax(energy, dim=-1)  # [B, 1, L]
        att_vec = torch.bmm(att_weight, values)  # [B, 1, L] * [B, L, d2] -> [B, 1, d2]

        return att_weight, att_vec


class BahdanauAttDecoder(nn.Module):
    '''
    Notes:
        1. Cannot do mini-batching?
    '''
    def __init__(self, args):
        super(BahdanauAttDecoder, self).__init__()
        self.args = args

        self.vocab_size = args.tgt_vocab_size
        self.word_dim = args.word_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.word_dim)
        self.init_embedding()
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(self.word_dim + self.lstm_hidden_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False, batch_first=True)

        self.att_module = AttentionModule(args)
        self.decoder_out = nn.Linear(self.lstm_hidden_dim * 3, self.vocab_size)

    def init_embedding(self):
        initrange = 0.5 / self.word_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, last_hidden, encoder_outputs, mask=None):
        # input_seqs: [B, 1] - feed inputs at one time step
        # last_hidden: ([1, B, d_de], [1, B, d_de]) for LSTM
        # encoder_outputs: [B, L, d_en]

        max_src_len = input_seqs.shape[1]
        embed_x = self.dropout(self.embeddings(input_seqs))  # [B, 1, d_emb]

        last_h_t = last_hidden[0].permute(1, 0, 2)  # [B, 1, d]
        att_weight, att_vec = self.att_module(last_h_t, encoder_outputs, mask=mask)  # [B, 1, L], [B, 1, d]

        embed_context = torch.cat((embed_x, att_vec), dim=-1)  # [B, 1, (d1+d2)]
        lstm_output, hidden = self.lstm(embed_context, last_hidden)  # [B, 1, d3]

        logits = self.decoder_out(torch.cat((embed_x, lstm_output, att_vec), dim=-1))  # [B, 1, V]
        # logits = logits.reshape(-1, self.vocab_size)
        return logits, hidden


class LuongAttDecoder(nn.Module):
    '''
    Global attention
    '''
    def __init__(self, args):
        super(LuongAttDecoder, self).__init__()
        self.args = args

        self.vocab_size = args.tgt_vocab_size
        self.word_dim = args.word_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.embeddings = nn.Embedding(self.vocab_size, self.word_dim)
        self.init_embedding()
        self.dropout = nn.Dropout(args.dropout)

        self.lstm = nn.LSTM(self.word_dim + self.lstm_hidden_dim, self.lstm_hidden_dim,
                            num_layers=1, bidirectional=False, batch_first=True)

        self.att_module = AttentionModule(args)
        self.decoder_out = nn.Linear(self.lstm_hidden_dim * 2, self.vocab_size)

    def init_embedding(self):
        initrange = 0.5 / self.word_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, last_context, last_hidden, encoder_outputs, mask=None):
        # one step at a time
        embed_x = self.dropout(self.embeddings(input_seqs))  # [B, 1, d1]

        embed_context = torch.cat((embed_x, last_context), dim=-1)
        lstm_output, hidden = self.lstm(embed_context, last_hidden)  # [B, L, d3]

        current_h_t = hidden[0].permute(1, 0, 2)  # [B, 1, d]
        att_weight, att_vec = self.att_module(current_h_t, encoder_outputs, mask=mask)  # [B, 1, L], [B, 1, d]

        lstm_context = torch.cat((lstm_output, att_vec), dim=-1)  # [B, L, (d3+d2)]
        logits = self.decoder_out(lstm_context)
        # logits = logits.reshape(-1, self.vocab_size)
        return logits, hidden, att_vec


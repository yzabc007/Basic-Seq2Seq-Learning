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

from EncoderDecoder import Encoder, Decoder, ContextDecoder, BahdanauAttDecoder, LuongAttDecoder


class SimpleSeq2Seq(nn.Module):
    def __init__(self, args):
        super(SimpleSeq2Seq, self).__init__()
        self.args = args
        self.src_word2id = args.src_word2id
        self.tgt_id2word = args.tgt_id2word
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_decode_len = args.max_decode_len

        self.device = args.device
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, x, teacher_forcing_ratio=0.5):
        input_src, src_length = x[0], x[1]
        input_tgt, tgt_length = x[2], x[3]
        # print(input_src.shape, src_length.shape)
        # print(input_tgt.shape, tgt_length.shape)
        # encoding
        encoder_output, encoder_hidden = self.encoder(input_src, src_length)

        # decoding
        # decoder_logits, decoder_hidden = self.decoder(input_tgt, tgt_length, encoder_hidden)

        recur_input = input_tgt[:, 0].unsqueeze(1)
        recur_hidden = encoder_hidden
        decoder_logits = []
        # decoder_logits = torch.zeros(input_tgt.shape[0], input_tgt.shape[1], self.tgt_vocab_size).to(self.device)
        for di in range(input_tgt.shape[1]):
            cur_logits, recur_hidden = self.decoder(recur_input, 1, recur_hidden)
            decoder_logits.append(cur_logits)
            cur_logits_flat = cur_logits.reshape(-1, self.tgt_vocab_size)
            teacher_force = np.random.random() < teacher_forcing_ratio
            recur_input = input_tgt[:, di].unsqueeze(1) if teacher_force else cur_logits_flat.topk(1)[1]

        decoder_logits = torch.cat(decoder_logits, dim=1)

        return decoder_logits

    def seq_decoding(self, src_sent, tgt_sent=None):
        with torch.no_grad():
            if isinstance(src_sent, str):
                src_sent = src_sent.split(' ')

            src_sent_idx = [self.src_word2id[w] if w in self.src_word2id else config.UNK_WORD for w in src_sent]
            src_sent_idx = [config.BOS_WORD] + src_sent_idx + [config.EOS_WORD]

            input_src = torch.tensor([src_sent_idx], device=self.device)
            src_length = torch.tensor([len(src_sent_idx)], device=self.device)

            # encoder_hidden = model.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(input_src, src_length)

            decoder_input = torch.tensor([[config.BOS_WORD]], device=self.device)
            decoder_hidden = encoder_hidden
            decoded_ids = []
            decoded_words = []
            for di in range(self.max_decode_len):
                decoder_logits, decoder_hidden = self.decoder(decoder_input, 1, decoder_hidden)
                # print(decoder_logits.shape)
                decoder_logits = decoder_logits.reshape(-1, self.tgt_vocab_size)
                topv, topi = decoder_logits.data.topk(1)
                if topi.item() == config.EOS_WORD:
                    decoded_ids.append(config.EOS_WORD)
                    decoded_words.append(config.EOS_WORD_S)
                    break
                else:
                    decoded_ids.append(topi.item())
                    decoded_words.append(self.tgt_id2word[topi.item()])
                decoder_input = topi

        return decoded_ids, decoded_words


class ContextSeq2Seq(nn.Module):
    def __init__(self, args):
        super(ContextSeq2Seq, self).__init__()
        self.args = args
        self.src_word2id = args.src_word2id
        self.tgt_id2word = args.tgt_id2word
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_decode_len = args.max_decode_len

        self.device = args.device
        self.teacher_forcing_ratio = args.teacher_forcing_ratio

        self.encoder = Encoder(args)
        self.decoder = ContextDecoder(args)

    def forward(self, x, teacher_forcing_ratio=0.5):
        input_src, src_length = x[0], x[1]
        input_tgt, tgt_length = x[2], x[3]
        # encoding
        encoder_output, encoder_hidden = self.encoder(input_src, src_length)

        # decoding - batch training currently
        context = encoder_hidden[0]
        # decoder_logits, decoder_hidden = self.decoder(input_tgt, context, tgt_length, encoder_hidden)

        recur_input = input_tgt[:, 0].unsqueeze(1)
        recur_hidden = encoder_hidden
        decoder_logits = []
        for di in range(input_tgt.shape[1]):
            cur_logits, recur_hidden = self.decoder(recur_input, context, 1, recur_hidden)
            decoder_logits.append(cur_logits)
            cur_logits_flat = cur_logits.reshape(-1, self.tgt_vocab_size)
            teacher_force = np.random.random() < teacher_forcing_ratio
            recur_input = input_tgt[:, di].unsqueeze(1) if teacher_force else cur_logits_flat.topk(1)[1]

        decoder_logits = torch.cat(decoder_logits, dim=1)

        return decoder_logits

    def seq_decoding(self, src_sent, tgt_sent=None):
        with torch.no_grad():
            if isinstance(src_sent, str):
                src_sent = src_sent.split(' ')

            src_sent_idx = [self.src_word2id[w] if w in self.src_word2id else config.UNK_WORD for w in src_sent]
            src_sent_idx = [config.BOS_WORD] + src_sent_idx + [config.EOS_WORD]

            input_src = torch.tensor([src_sent_idx], device=self.device)
            src_length = torch.tensor([len(src_sent_idx)], device=self.device)

            # encoder_hidden = model.encoder.initHidden()
            encoder_output, encoder_hidden = self.encoder(input_src, src_length)

            decoder_input = torch.tensor([[config.BOS_WORD]], device=self.device)
            decoder_hidden = encoder_hidden
            context = encoder_hidden[0]
            decoded_ids = []
            decoded_words = []
            for di in range(self.max_decode_len):
                decoder_logits, decoder_hidden = self.decoder(decoder_input, context, 1, decoder_hidden)
                # print(decoder_logits.shape)
                decoder_logits = decoder_logits.reshape(-1, self.tgt_vocab_size)
                topv, topi = decoder_logits.data.topk(1)
                if topi.item() == config.EOS_WORD:
                    decoded_ids.append(config.EOS_WORD)
                    decoded_words.append(config.EOS_WORD_S)
                    break
                else:
                    decoded_ids.append(topi.item())
                    decoded_words.append(self.tgt_id2word[topi.item()])
                decoder_input = topi

        return decoded_ids, decoded_words


class BahdanauAttSeq2Seq(nn.Module):
    '''
    Basic Attention-based Seq2seq model
    '''
    def __init__(self, args):
        super(BahdanauAttSeq2Seq, self).__init__()
        self.args = args
        self.src_word2id = args.src_word2id
        self.tgt_id2word = args.tgt_id2word
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_decode_len = args.max_decode_len

        self.device = args.device

        self.encoder = Encoder(args)
        self.decoder = BahdanauAttDecoder(args)

    def forward(self, x, teacher_forcing_ratio=0.5):
        input_src, src_length = x[0], x[1]
        input_tgt, tgt_length = x[2], x[3]
        # print(input_src.shape, src_length.shape)
        # print(input_tgt.shape, tgt_length.shape)
        # encoding
        encoder_outputs, encoder_hidden = self.encoder(input_src, src_length)

        # decoding
        recur_input = input_tgt[:, 0].unsqueeze(1)
        recur_hidden = encoder_hidden
        decoder_logits = []
        for di in range(input_tgt.shape[1]):
            cur_logits, recur_hidden = self.decoder(recur_input, recur_hidden, encoder_outputs)
            decoder_logits.append(cur_logits)
            cur_logits_flat = cur_logits.reshape(-1, self.tgt_vocab_size)
            teacher_force = np.random.random() < teacher_forcing_ratio
            recur_input = input_tgt[:, di].unsqueeze(1) if teacher_force else cur_logits_flat.topk(1)[1]

        decoder_logits = torch.cat(decoder_logits, dim=1)

        return decoder_logits

    def seq_decoding(self, src_sent, tgt_sent=None):
        with torch.no_grad():
            if isinstance(src_sent, str):
                src_sent = src_sent.split(' ')

            src_sent_idx = [self.src_word2id[w] if w in self.src_word2id else config.UNK_WORD for w in src_sent]
            src_sent_idx = [config.BOS_WORD] + src_sent_idx + [config.EOS_WORD]

            input_src = torch.tensor([src_sent_idx], device=self.device)
            src_length = torch.tensor([len(src_sent_idx)], device=self.device)

            # encoder_hidden = model.encoder.initHidden()
            encoder_outputs, encoder_hidden = self.encoder(input_src, src_length)

            decoder_input = torch.tensor([[config.BOS_WORD]], device=self.device)
            decoder_hidden = encoder_hidden
            decoded_ids = []
            decoded_words = []
            for di in range(self.max_decode_len):
                decoder_logits, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # print(decoder_logits.shape)
                decoder_logits = decoder_logits.reshape(-1, self.tgt_vocab_size)
                topv, topi = decoder_logits.data.topk(1)
                if topi.item() == config.EOS_WORD:
                    decoded_ids.append(config.EOS_WORD)
                    decoded_words.append(config.EOS_WORD_S)
                    break
                else:
                    decoded_ids.append(topi.item())
                    decoded_words.append(self.tgt_id2word[topi.item()])
                decoder_input = topi

        return decoded_ids, decoded_words


class LuongAttSeq2Seq(nn.Module):
    '''
    Basic Attention-based Seq2seq model
    '''
    def __init__(self, args):
        super(LuongAttSeq2Seq, self).__init__()
        self.args = args
        self.src_word2id = args.src_word2id
        self.tgt_id2word = args.tgt_id2word
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_decode_len = args.max_decode_len
        self.lstm_hidden_dim = args.lstm_hidden_dim

        self.device = args.device

        self.encoder = Encoder(args)
        self.decoder = LuongAttDecoder(args)

    def forward(self, x, teacher_forcing_ratio=0.5):
        input_src, src_length = x[0], x[1]
        input_tgt, tgt_length = x[2], x[3]
        # print(input_src.shape, src_length.shape)
        # print(input_tgt.shape, tgt_length.shape)
        # encoding
        batch_size = input_tgt.shape[0]
        encoder_outputs, encoder_hidden = self.encoder(input_src, src_length)

        # decoding
        recur_input = input_tgt[:, 0].unsqueeze(1)
        recur_hidden = encoder_hidden
        recur_context = torch.zeros((batch_size, 1, self.lstm_hidden_dim), device=self.device)
        decoder_logits = []
        for di in range(input_tgt.shape[1]):
            cur_logits, recur_hidden, recur_context = self.decoder(recur_input,
                                                                   recur_context,
                                                                   recur_hidden,
                                                                   encoder_outputs)
            decoder_logits.append(cur_logits)
            cur_logits_flat = cur_logits.reshape(-1, self.tgt_vocab_size)
            teacher_force = np.random.random() < teacher_forcing_ratio
            recur_input = input_tgt[:, di].unsqueeze(1) if teacher_force else cur_logits_flat.topk(1)[1]

        decoder_logits = torch.cat(decoder_logits, dim=1)

        return decoder_logits

    def seq_decoding(self, src_sent, tgt_sent=None):
        with torch.no_grad():
            if isinstance(src_sent, str):
                src_sent = src_sent.split(' ')

            src_sent_idx = [self.src_word2id[w] if w in self.src_word2id else config.UNK_WORD for w in src_sent]
            src_sent_idx = [config.BOS_WORD] + src_sent_idx + [config.EOS_WORD]

            input_src = torch.tensor([src_sent_idx], device=self.device)
            src_length = torch.tensor([len(src_sent_idx)], device=self.device)

            # encoder_hidden = model.encoder.initHidden()
            encoder_outputs, encoder_hidden = self.encoder(input_src, src_length)

            decoder_input = torch.tensor([[config.BOS_WORD]], device=self.device)
            decoder_hidden = encoder_hidden
            decoder_context = torch.zeros((1, 1, self.lstm_hidden_dim), device=self.device)
            decoded_ids = []
            decoded_words = []
            for di in range(self.max_decode_len):
                decoder_logits, decoder_hidden, decoder_context = self.decoder(decoder_input,
                                                                               decoder_context,
                                                                               decoder_hidden,
                                                                               encoder_outputs)
                # print(decoder_logits.shape)
                decoder_logits = decoder_logits.reshape(-1, self.tgt_vocab_size)
                topv, topi = decoder_logits.data.topk(1)
                if topi.item() == config.EOS_WORD:
                    decoded_ids.append(config.EOS_WORD)
                    decoded_words.append(config.EOS_WORD_S)
                    break
                else:
                    decoded_ids.append(topi.item())
                    decoded_words.append(self.tgt_id2word[topi.item()])
                decoder_input = topi

        return decoded_ids, decoded_words

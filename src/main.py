import os
import sys
import pickle
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import config
import utils
import loader
import train_utils
import Seq2Seq


def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]


def main():
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--random_seed', type=float, default=42)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--max_decode_len', type=int, default=100)
    parser.add_argument('--att_method', type=str, default='concat')
    parser.add_argument('--seq_model', type=str, default='simple')

    # I/O parameters
    parser.add_argument('--train_dir', type=str, default='./eng-fra.txt')
    parser.add_argument('--word_embed_file', type=str,
                        default='/home/wang.9215/medical_phrases/pretrain_embeddings/glove.6B.300d.txt')
    parser.add_argument('--save_best', type='bool', default=False, help='save model in the best epoch or not')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='save model in the best epoch or not')
    parser.add_argument('--save_interval', type=int, default=5, help='intervals for saving models')

    # model parameters
    parser.add_argument('--use_pretrain_embed', type='bool', default=False)
    parser.add_argument('--word_dim', type=int, default=256)
    parser.add_argument('--lstm_hidden_dim', type=int, default=256)

    # optim parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000, help="number of epochs for training")
    parser.add_argument('--log_interval', type=int, default=100, help='step interval for log')
    parser.add_argument('--test_interval', type=int, default=10, help='epoch interval for testing')
    parser.add_argument('--early_stop_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--min_epochs', type=int, default=50, help='minimum number of epochs')
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--metric', type=str, default='map', help='mrr or map')
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    print('args: ', args)

    print('********Key parameters:******')
    print('Use GPU? {0}'.format(torch.cuda.is_available()))
    # print('Model Parameters: ')
    print('*****************************')

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # load the original data
    data, src_vocab, tgt_vocab = loader.txt_loader(args.train_dir)

    _, src_word2id, src_id2word = loader.raw_word_mapping(src_vocab)
    args.src_vocab_size = len(src_word2id) + 1
    args.src_word2id = src_word2id
    args.src_id2word = src_id2word

    _, tgt_word2id, tgt_id2word = loader.raw_word_mapping(tgt_vocab)
    args.tgt_vocab_size = len(tgt_word2id) + 1
    args.tgt_word2id = tgt_word2id
    args.tgt_id2word = tgt_id2word

    np.random.shuffle(data)
    train_ratio = 0.8
    train_data = data[:int(len(data) * train_ratio)]
    test_data = data[int(len(data) * train_ratio)::]
    # convert original data to digits
    train_idx_data = loader.make_idx_data(train_data, args)
    test_idx_data = loader.make_idx_data(test_data, args)

    print(train_idx_data[0])

    # global parameters
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model, initialize parameters
    # model = Seq2Seq.SimpleSeq2Seq(args).to(args.device)
    # model = Seq2Seq.ContextSeq2Seq(args).to(args.device)
    # model = Seq2Seq.BahdanauAttSeq2Seq(args).to(args.device)
    model = Seq2Seq.LuongAttSeq2Seq(args).to(args.device)
    # if args.cuda:
    #     model = model.cuda()
    print(model)
    # print([name for name, p in model.named_parameters()])

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_WORD)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loss = 0
    train_logits = []
    train_labels = []
    num_batches = len(train_idx_data) // args.batch_size
    print('Begin trainning...')
    for epoch in range(args.num_epochs):
        model.train()
        steps = 0
        np.random.shuffle(train_idx_data)
        for i in range(num_batches):
            train_batch = train_idx_data[i * args.batch_size: (i + 1) * args.batch_size]
            if i == num_batches - 1:
                train_batch = train_idx_data[i * args.batch_size::]

            list_xs, labels = loader.batch_processing(train_batch, args)

            labels = torch.tensor(labels, dtype=torch.long, device=args.device)  # (B * L)
            list_xs = [torch.tensor(x, device=args.device) for x in list_xs]
            # if args.cuda:
            #     labels = labels.cuda()
            #     list_xs = [x.cuda() for x in list_xs]

            optimizer.zero_grad()

            logits = model(list_xs, args.teacher_forcing_ratio)  # (B * L) * V
            # loss = utils.masked_cross_entropy(logits, labels)
            loss = criterion(logits.reshape(-1, args.tgt_vocab_size), labels.reshape(-1))
            train_loss += loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
                # logits = logits.cpu().detach().data.numpy()
                # labels = labels.cpu().detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            train_logits.append(logits)
            train_labels.append(labels)

            # evaluation
            steps += 1
            if steps % args.log_interval == 0:
                # cur_logits = np.concatenate(train_logits, axis=0)
                # cur_labels = np.concatenate(train_labels, axis=0)
                cur_train_bleu = utils.cal_bleu_score(train_logits, train_labels, config.EOS_WORD)
                print("Epoch-{0}, steps-{1}: Train Loss - {2:.5}, Train BLEU - {3:.5}".
                      format(epoch, steps, train_loss / len(train_batch), cur_train_bleu))

                train_loss = 0
                train_logits = []
                train_labels = []

        # utils.adjust_learning_rate(optimizer, args.learning_rate / (1 + (epoch + 1) * args.lr_decay))

        if epoch == 0: continue

        if epoch % args.test_interval == 0:
            test_bleu = train_utils.dataset_eval(model, test_idx_data, args)
            print("Epoch-{0}: Test BLEU: {1:.5}".format(epoch, test_bleu))

            # test_case = np.random.choice(test_idx_data)
            idx = np.random.randint(0, len(test_idx_data))
            print(' '.join(test_data[idx][0]))
            print(' '.join(test_data[idx][1]))
            decode_ids, decode_words = model.seq_decoding(test_data[idx][0])
            # decode_words = train_utils.instance_eval(model, test_data[idx][0], test_data[idx][1], args, seq_decode=True)
            print(decode_words)

        # if epoch % args.test_interval == 0:
        #     # if args.logging:
        #     #     args.log_name = '../logs/deep_predition_val_logs_{0}.txt'.format(epoch)
        #     all_dev = train_utils.evaluation(dev_idx, model, criterion, args)
        #     print("Epoch-{0}: All Dev {1}: {2:.5}".format(epoch, args.metric.upper(), all_dev))
        #
        #     if all_dev > best_on_dev:
        #         print(datetime.now().strftime("%m/%d/%Y %X"))
        #         best_on_dev = all_dev
        #         last_epoch = epoch
        #
        #     all_iv = train_utils.evaluation(iv_test_idx, model, criterion, args)
        #     print("--- Testing: All IV Test {0}: {1:.5}".format(args.metric.upper(), all_iv))
        #
        #     if args.save_best:
        #         utils.save(model, args.save_dir, 'best', epoch)
        #
        #     else:
        #         if epoch - last_epoch > args.early_stop_epochs and epoch > args.min_epochs:
        #             print('Early stop at {0} epoch.'.format(epoch))
        #             break
        #
        # elif epoch % args.save_interval == 0:
        #     utils.save(model, args.save_dir, 'snapshot', epoch)

    return


if __name__ == '__main__':
    main()

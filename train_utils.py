import os
import sys

import torch

import utils
import config
import loader


def instance_eval(model, src_sent, tgt_sent, args, seq_decode=False):
    with torch.no_grad():
        model.eval()
        if isinstance(src_sent, str):
            src_sent = src_sent.split(' ')

        src_data = [config.BOS_WORD] + src_sent + [config.EOS_WORD]
        src_sent_idx = [args.src_word2id[w if w in args.src_word2id else config.UNK_WORD] for w in src_data]
        input_src = torch.tensor([src_sent_idx], device=args.device)
        src_length = torch.tensor([len(src_sent_idx)], device=args.device)
        # input_src = utils.tesnoring([src_sent_idx], args.cuda)
        # src_length = utils.tesnoring([len(src_sent_idx)], args.cuda)
        # print(input_src, src_length)

        # encoder_hidden = model.encoder.initHidden()
        encoder_output, encoder_hidden = model.encoder(input_src, src_length)

        '''
        # if not seq_decode:
        if isinstance(tgt_sent, str):
            tgt_sent = tgt_sent.split(' ')

        tgt_data = [config.BOS_WORD] + tgt_sent
        tgt_sent_idx = [args.tgt_word2id[w if w in args.tgt_word2id else config.UNK_WORD] for w in tgt_data]
        input_tgt = torch.tensor([tgt_sent_idx], device=args.device)
        tgt_length = torch.tensor([len(tgt_sent_idx)], device=args.device)
        # print(input_tgt, tgt_length)

        decoder_hidden = encoder_hidden
        decoder_logits, decode_hidden = model.decoder(input_tgt, tgt_length, decoder_hidden)
        decoder_logits = decoder_logits.view(-1, args.tgt_vocab_size)
        decoded_words = [args.tgt_id2word[x.item()] for x in torch.argmax(decoder_logits, dim=-1)]
        print('>>>>>', decoded_words)

        # decoder_hidden = encoder_hidden
        # decoded_words = []
        # for di in range(len(tgt_sent_idx)):
        #     decoder_input = input_tgt[:, di].unsqueeze(1).detach()
        #     decoder_logits, decoder_hidden = model.decoder(decoder_input, 1, decoder_hidden)
        #     decoder_logits = decoder_logits.reshape(-1, args.tgt_vocab_size)
        #     decoded_words.append(args.tgt_id2word[torch.argmax(decoder_logits, dim=-1).item()])
        '''

        decoder_input = torch.tensor([[args.tgt_word2id[config.BOS_WORD]]], device=args.device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        for di in range(100):
            decoder_logits, decoder_hidden = model.decoder(decoder_input, 1, decoder_hidden)
            # print(decoder_logits.shape)
            decoder_logits = decoder_logits.reshape(-1, args.tgt_vocab_size)
            topv, topi = decoder_logits.data.topk(1)
            if topi.item() == args.tgt_word2id[config.EOS_WORD]:
                decoded_words.append(config.EOS_WORD)
                break
            else:
                decoded_words.append(args.tgt_id2word[topi.item()])
            decoder_input = topi

        # print('-----', decoded_words)

    return decoded_words


def dataset_eval(model, test_data, args):
    with torch.no_grad():
        model.eval()
        test_logits = []
        test_labels = []
        for data in test_data:

            list_xs, labels = loader.batch_processing([data], args)

            labels = torch.tensor(labels, dtype=torch.long, device=args.device)  # (B * L)
            list_xs = [torch.tensor(x, device=args.device) for x in list_xs]
            # if args.cuda:
            #     labels = labels.cuda()
            #     list_xs = [x.cuda() for x in list_xs]

            # decoded_ids, decoded_words = model.seq_decoding(list_xs[0])
            logits = model(list_xs, 0)  # (B * L) * V

            if args.cuda:
                logits = logits.to('cpu').detach().data.numpy()
                labels = labels.to('cpu').detach().data.numpy()
                # logits = logits.cpu().detach().data.numpy()
                # labels = labels.cpu().detach().data.numpy()
            else:
                logits = logits.detach().data.numpy()
                labels = labels.detach().data.numpy()

            # print(logits.shape)
            test_logits.append(logits)
            test_labels.append(labels)

        test_bleu = utils.cal_bleu_score(test_logits, test_labels, config.EOS_WORD)

    return test_bleu
# Basic-Seq2Seq-Lerning
Basic seq2seq model including simplest encoder &amp; decoder and attention-based ones

This repository implements the "most" basic seq2seq learning as one small step of my own project.

Basically, it constains the essential ideas of the following papers:

**SimpleSeq2Seq** model: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- Simplest rnn encoder and rnn decoder

**ContextSeq2Seq** model: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- Adding context vector to each step of decoder (rnn inputs & classifier inputs)

**BahdanauAttSeq2Seq** model: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Bahdanau Attention mechanism: first calculating attention weight and vector with previous hidden state of the decoder and the hidden states of the encoder; then feeding all info to the decoder rnn and doing classification by current hidden state of the decoder

**LuongAttSeq2Seq** model: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- Global attention mechanism: first getting hidden state of the decoder and calculate attention weight and vector with current hidden state of the decoder and the hidden states of the encoder; then feeding all info to the classifier

This implementation features the following functionalities:
- Easy to use and understand code framework with simple pre-processing and decoding 
- Mini-batching (although the decoder goes one step at one time)

TODO:
- ~~Adding mask for attention~~
- Making the passing of dimension para more clear
- Comparing the performance with reported results of papers in formal datasets
- Beam searching
- Implementing "Attention is all you need".

Some resources I refer to during the implementation:
> https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

> https://github.com/bentrevett/pytorch-seq2seq

> https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation

> https://github.com/IBM/pytorch-seq2seq

> https://github.com/MaximumEntropy/Seq2Seq-PyTorch

> https://github.com/pytorch/tutorials/tree/master/intermediate_source


Please feel free to point them out if there are errors whether in the implementation or my understanding of papers. Thanks.

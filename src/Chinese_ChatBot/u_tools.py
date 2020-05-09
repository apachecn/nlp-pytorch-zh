#!/usr/bin/python
# coding: utf-8
import os
import re
import unicodedata
import torch
import torch.nn as nn
from torch import optim
from u_class import Voc, EncoderRNN, LuongAttnDecoderRNN


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!? ])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z.!? \u4E00-\u9FA5]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    # '咋死 ? ? ?红烧还是爆炒dddd' > '咋 死   ?   ?   ? 红 烧 还 是 爆 炒 d d d d'
    s = " ".join(list(s))
    return s


def load_model(loadFilename, voc, cp_start_iteration, attn_model, hidden_size, encoder_n_layers, decoder_n_layers, 
        dropout, learning_rate, decoder_learning_ratio):

    # Load model if a loadFilename is provided
    if os.path.exists(loadFilename):
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        cp_start_iteration = checkpoint['iteration']
        encoder_sd = checkpoint['state_dict_en']
        decoder_sd = checkpoint['state_dict_de']
        encoder_optimizer_sd = checkpoint['state_dict_en_opt']
        decoder_optimizer_sd = checkpoint['state_dict_de_opt']
        # loss = checkpoint['loss']
        # voc.__dict__ = checkpoint['voc_dict']
        embedding_sd = checkpoint['embedding']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if os.path.exists(loadFilename):
        embedding.load_state_dict(embedding_sd)

    print('Initialize encoder & decoder models')
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if os.path.exists(loadFilename):
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    print('Models built and ready to go!')
    return cp_start_iteration, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding

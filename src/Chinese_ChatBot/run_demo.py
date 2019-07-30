#!/usr/bin/python
# coding: utf-8
import os
import re
import unicodedata
import torch
import torch.nn as nn
from u_tools import normalizeString, load_model
from u_class import Voc, GreedySearchDecoder


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 50  # Maximum sentence length to consider


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            # print('Bot:', ' '.join(output_words))
            print('Bot:', ''.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == "__main__":
    global device, corpus_name
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    corpus_name = "Chinese_ChatBot"


    # Configure models
    attn_model = 'dot'
    #attn_model = 'general'
    #attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    cp_start_iteration = 0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = 8000

    voc = Voc(corpus_name)
    loadFilename = "data/save/cb_model/%s/2-2_500/%s_checkpoint.tar" % (corpus_name, n_iteration)
    if os.path.exists(loadFilename):
        checkpoint = torch.load(loadFilename)
        voc.__dict__ = checkpoint['voc_dict']

    cp_start_iteration, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding = load_model(loadFilename, voc, cp_start_iteration, attn_model, hidden_size, encoder_n_layers, decoder_n_layers, dropout, learning_rate, decoder_learning_ratio)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder, device)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(encoder, decoder, searcher, voc)

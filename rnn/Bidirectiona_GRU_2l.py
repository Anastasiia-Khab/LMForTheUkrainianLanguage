#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Code RNnLM from: https://github.com/florijanstamenkovic/PytorchRnnLM/blob/master/main.py
Bidirectional implementation: https://github.com/Anastasiia-Khab

Trains Bidirectional Language Model GRU 
2 layer, 
0.4 clipping, 
2 dropout, 

Optimizer: Adam

Train - Korresponden+Ukr.fiction dataset
Test - Brown Ukrainian Corpus
(GeForce RTX 2080 Ti (11GB RAM))  
"""

from argparse import ArgumentParser
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_loader
from vocab import Vocab
from log_timer import LogTimer
from datetime import datetime
from utilities import *
logging.basicConfig(filename='logging/GRUlogging_2lAdam_bidirectional.log',level=logging.DEBUG)

class RnnLm(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers, tied, dropout):
        super(RnnLm, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout,bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        logging.debug("Net:\n%r", self)
        print("Net:\n%r", self)

    def get_embedded(self, word_indexes):
        if self.tied:
            return self.fc.weight.index_select(0, word_indexes)
        else:
            return self.embedding(word_indexes)

    def forward(self, packed_sents):
        """ Takes a PackedSequence of sentences tokens that has T tokens
        belonging to vocabulary V. Outputs predicted log-probabilities
        for the token following the one that's input in a tensor shaped
        (T, |V|).
        """
        embedded_sents = nn.utils.rnn.PackedSequence(
            self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, out_packed_hidden = self.gru(embedded_sents)
        #print("packed output ", out_packed_sequence)
        padded=nn.utils.rnn.pad_packed_sequence(out_packed_sequence, batch_first=True, padding_value=100, total_length=None)
        forward_output = [padded[0].data[i][:padded[1].data[i]][:,:embedding_dim][:-2] for i in range(len(padded[1].data))]
        backward_output = [padded[0].data[i][:padded[1].data[i]][:,embedding_dim:][2:] for i in range(len(padded[1].data))] 
        #print(forward_output)
        #print(backward_output)
        output=[torch.cat((o1, o2), dim=-1) for o1,o2 in zip(forward_output,forward_output)]
        out_packed_sequence_cat=nn.utils.rnn.pack_sequence(output)
        out1 = self.fc1(out_packed_sequence_cat.data)
        out = self.fc(out1)
        return F.log_softmax(out, dim=1)


def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


def step(model, sents, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:] for s in sents])
    #print('x',x)
    #y = torch.cat((y_forward, y_backward), dim=-1)
    y = nn.utils.rnn.pack_sequence([s[1:-1] for s in sents])
    #print('y',y)
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    #print("out ", out.shape)
    loss = F.nll_loss(out, y.data)
    return out, loss, y



def train_epoch(data, model, optimizer, batch_size, device):
    """ Trains a single epoch of the given model. """
    model.train()
    log_timer = LogTimer(100)
    for batch_ind, sents in enumerate(batches(data, batch_size)):
        model.zero_grad()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        out, loss, y = step(model, sents, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.2)
        try:
            optimizer.step()
        except Exception as e:
            logging.fatal(e, exc_info=True)
        if log_timer() or batch_ind == 0:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            print("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)


def evaluate(data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, device)
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)


embedding_dim=300
untied=True
gru_hidden=300
gru_layers=2
gru_dropout=0.4 #"The amount of dropout in GRU layers"
epochs=4
batch_size=50
lr=0.001
#no_cuda=True

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:3")
torch.cuda.set_device(device)
logging.debug ("Current device %s ", torch.cuda.current_device())
print("Current device %s ", torch.cuda.current_device())

data_path = "korr_ukrlib_data.pkl"
test_path = "brown_test_data.pkl"

logging.debug ("Loading data ...")
print("Loading data ...")
data = load_training_data(data_path)
data_test = load_training_data(test_path)
print("DONE loading data ...")
logging.debug ("DONE loading data ...")

vocab = Vocab()
vocab._ind_to_tok=data['index_to_word']
vocab._tok_to_ind=data['word_to_index']

test_split=0.04
num_sentences = data["num_sentences"]
num_valid_samples = math.ceil(num_sentences * test_split)

#num_test_samples=1000

train_data = [sentence for sentence in data['x_train'][num_valid_samples:] if len(sentence) > 2]
valid_data = [sentence for sentence in data['x_train'][:num_valid_samples] if len(sentence) > 2]
test_data = [sentence for sentence in data_test['x_test'] if len(sentence) > 2]

model = RnnLm(len(vocab), embedding_dim,
                  gru_hidden, gru_layers,
                  not untied, gru_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
torch.cuda.empty_cache()
best_valid_pp=10000000

for epoch_ind in range(epochs):
    logging.debug("START EPOCH now = %s", datetime.now())
    print("START EPOCH now = ", datetime.now())
    logging.debug("Training epoch %d", epoch_ind)
    print("Training epoch %d", epoch_ind)
    train_epoch(train_data, model, optimizer, batch_size, device)
    valid_pp=evaluate(valid_data, model, batch_size, device)
    logging.debug("Validation perplexity: %.1f",valid_pp)
    print("Validation perplexity: %.1f",valid_pp)
    test_pp=evaluate(test_data, model, batch_size, device)
    logging.debug ("Test perplexity: %.1f",test_pp)
    print("Test perplexity: %.1f", test_pp)
    if valid_pp<best_valid_pp:
        logging.debug("BEST result")
        print("BEST result")
    PATH="simpleGRUmodel_2lAdam_40_bidirectional.pt"
    torch.save({
               'epoch': epoch_ind,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()
                 }, PATH)
    logging.debug("END EPOCH now = %s", datetime.now())
    print("END EPOCH now = ", datetime.now())

PATH="simpleGRUmodel_2lAdam_40_bidirectional.pt"
model = RnnLm(len(vocab), embedding_dim,
                  gru_hidden, gru_layers,
                  not untied, gru_dropout).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_ind = checkpoint['epoch']
test_pp=evaluate(test_data, model, batch_size, device)
print("Test perplexity after the loading: %.1f", test_pp)
logging.debug("Test perplexity after the loading: %.1f", test_pp)






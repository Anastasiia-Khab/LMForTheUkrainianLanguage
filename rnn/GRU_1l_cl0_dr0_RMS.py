#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Code from: https://github.com/florijanstamenkovic/PytorchRnnLM/blob/master/main.py

Trains Language Model GRU 
1 layer, 
0 clipping, 
0 dropout, 

Optimizer: RMSProp

Train - Korresponden+Ukr.fiction dataset
Test - Brown Ukrainian Corpus
(GeForce RTX 2080 Ti (11GB RAM))
Size: 1380Mb
NParams: 180 841 800
Time1epoch:\approx7:40  

Training results (test):
1: 1123.3
2: 878.8
3: 835.6
4: 793.2
5: 666.9
6: 676.1
7: 682.5
8: inf
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
logging.basicConfig(filename='logging/GRU_1l_cl0_dr0_RMS.log',level=logging.DEBUG)

class RnnLm(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, gru_layers, tied, dropout):
        super(RnnLm, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)
        logging.debug("Net:\n%r", self)
        print("Net:\n%r", self)

    def get_embedded(self, word_indexes):
        if self.tied:
            return self.fc1.weight.index_select(0, word_indexes)
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
        out_packed_sequence, _ = self.gru(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
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
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y


def train_epoch(data, model, optimizer, batch_size, device):
    """ Trains a single epoch of the given model. """
    torch.cuda.empty_cache()
    model.train()
    log_timer = LogTimer(100)
    for batch_ind, sents in enumerate(batches(data, batch_size)):
        model.zero_grad()
        out, loss, y = step(model, sents, device)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),0.25)
        optimizer.step()
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
    torch.cuda.empty_cache()
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
gru_layers=1
gru_dropout=0.0 #"The amount of dropout in GRU layers"
epochs=3
batch_size=40
lr=0.001
#no_cuda=True

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
torch.cuda.set_device(device)
logging.debug ("Current device %s ", torch.cuda.current_device())
print("Current device %s ", torch.cuda.current_device())

data_path = "../data/korr_ukrlib_data.pkl"
test_path = "../data/brown_test_data.pkl"

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
random.shuffle(data['x_train'])
train_data = [sentence for sentence in data['x_train'][num_valid_samples:] if len(sentence) > 2]
valid_data = [sentence for sentence in data['x_train'][:num_valid_samples] if len(sentence) > 2]
test_data = [sentence for sentence in data_test['x_test'] if len(sentence) > 2]

torch.cuda.empty_cache()
PATH="bestsimpleGRUmodel.pt"
model = RnnLm(len(vocab), embedding_dim,
                  gru_hidden, gru_layers,
                  not untied, gru_dropout).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_ind_loaded = checkpoint['epoch']
epoch_ind = checkpoint['epoch']
torch.cuda.empty_cache()
test_pp=evaluate(test_data, model, batch_size, device)
print("Test perplexity after the loading: %.1f", test_pp)
logging.debug ("Test perplexity: %.1f",test_pp)
best_valid_pp=10000

for ind in range(5):
    epoch_ind+=ind
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
        PATH="bestsimpleGRUmodel.pt"
        torch.save({
                   'epoch': epoch_ind,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()
                     }, PATH)
        best_valid_pp=valid_pp
        logging.debug("BEST result")
        print("BEST result")
    PATH="simpleGRUmodel.pt"
    torch.save({
               'epoch': epoch_ind,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()
                 }, PATH)
    logging.debug("END EPOCH now = %s", datetime.now())
    print("END EPOCH now = ", datetime.now())

PATH="simpleGRUmodel.pt"
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







#!/usr/bin/python3
# -*- coding: UTF-8 -*-

"""
Code from: https://github.com/florijanstamenkovic/PytorchRnnLM/blob/master/main.py

Trains Language Model LSTM 
3 layer, 
0.2 clipping, 
20% dropout, 
Optimizer: Adam

Train - Korresponden+Ukr.fiction dataset
Test - Brown Ukrainian Corpus
(TITAN X(Pascal) (12GB RAM))
Size: 2815Mb
NParams: 245 912 000
Time1epoch:25:30   
1: 331.5
2: 298.2
3: 287.4
4: 279.2
5: 269.4
6: 270.6
7: 268.5
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
from torch.optim.lr_scheduler import StepLR
logging.basicConfig(filename='logging/LSTMlogging_3lAdam_optim.log',level=logging.DEBUG)

class RnnLm(nn.Module):
    """ A language model RNN with lstm layer(s). """

    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, lstm_layers, tied, dropout):
        super(RnnLm, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers,
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
        out_packed_sequence, _ = self.lstm(embedded_sents)
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
        x, y = x.to(device=device), y.to(device=device)
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
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        out, loss, y = step(model, sents, device)
        torch.cuda.empty_cache()
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

def count_parameters(model):
    for name, p in model.named_parameters():
        logging.debug ("Parameter %s ", str(name))
        logging.debug ("Parameter shape %s ", str(p.shape))
        logging.debug ("Parameter %s ", str(p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


embedding_dim=300
untied=True
lstm_hidden=500
lstm_layers=3
lstm_dropout=0.2 #"The amount of dropout in LSTM layers"
epochs=10
batch_size=40
lr=0.001
#no_cuda=True

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:1")
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

train_data = [sentence for sentence in data['x_train'][:-num_valid_samples] if len(sentence) > 2]
valid_data = [sentence for sentence in data['x_train'][-num_valid_samples:] if len(sentence) > 2]
test_data = [sentence for sentence in data_test['x_test'] if len(sentence) > 2]


torch.cuda.empty_cache()
#PATH="bestsimpleLSTMmodel_3lAdam.pt"
model = RnnLm(len(vocab), embedding_dim,
                  lstm_hidden, lstm_layers,
                  not untied, lstm_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
#checkpoint = torch.load(PATH)
#model.load_state_dict(checkpoint['model_state_dict'])
#model.to(device)
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch_ind_loaded = checkpoint['epoch']
#epoch_ind = checkpoint['epoch']
epoch_ind = -1
#torch.cuda.empty_cache()
#test_pp=evaluate(test_data, model, batch_size, device)
#print("Test perplexity after the loading: ", test_pp)
#logging.debug ("Test perplexity: %.1f",test_pp)
best_valid_pp=10000
torch.cuda.empty_cache()

for ind in range(epochs):
    epoch_ind+=1
    print(count_parameters(model))
    logging.debug("Number parameters %s", str(count_parameters(model)))
    print(scheduler.get_lr())
    logging.debug("Scheduler.get_lr() %s", str(scheduler.get_lr()))
    random.shuffle(data['x_train'])
    train_data = [sentence for sentence in data['x_train'][num_valid_samples:] if len(sentence) > 2]
    valid_data = [sentence for sentence in data['x_train'][:num_valid_samples] if len(sentence) > 2]
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
        PATH="bestsimpleLSTMmodel_3lAdam_optim.pt"
        torch.save({
                   'epoch': epoch_ind,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()
                     }, PATH)
        best_test_pp=test_pp
        logging.debug("BEST result")
        print("BEST result!!!Saved!")
    scheduler.step()
    logging.debug("END EPOCH now = %s", datetime.now())
    print("END EPOCH now = ", datetime.now())

PATH="bestsimpleLSTMmodel_3lAdam_optim.pt"
model = RnnLm(len(vocab), embedding_dim,
                  lstm_hidden, lstm_layers,
                  not untied, lstm_dropout).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=lr)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch_ind = checkpoint['epoch']
test_pp=evaluate(test_data, model, batch_size, device)
print("Test perplexity after the loading: %.1f", test_pp)
logging.debug("Test perplexity after the loading: %.1f", test_pp)






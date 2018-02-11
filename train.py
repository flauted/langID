#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import model
import dataset
import utils
import time
import sys
import argparse


use_cuda = torch.cuda.is_available()


def train(train_loader, batch_size, vocab_size, bidirectional=None):
    """Run the training loop.

    Parameters
    ----------
    train_loader : DataLoader
        The training dataset in -Loader format.
    batch_size : int
    vocab_size : int

    Returns
    -------
    LSTMClassifier
        The trained LSTM.

    """
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 128
    if bidirectional:
        LSTM = model.BiLSTMClassifier(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            vocab_size,
            batch_size)
    else:
        LSTM = model.LSTMClassifier(
            EMBEDDING_DIM,
            HIDDEN_DIM,
            vocab_size,
            batch_size)
    if use_cuda:
        LSTM = LSTM.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(LSTM.parameters(), lr=0.1)

    print_every = 1
    n_epochs = 1

    for epoch in range(1, n_epochs+1):
        start = time.time()
        epoch_loss = 0
        for data in train_loader:
            if use_cuda:
                sentence = Variable(data["sentence"].cuda())
                label = Variable(data["language"].cuda())
            else:
                sentence = Variable(data["sentence"])
                label = Variable(data["language"])
            LSTM.zero_grad()
            LSTM.hidden = LSTM.init_hidden()
            pred = LSTM(sentence)
            loss = loss_function(pred, label)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

        if epoch % print_every == 0:
            loss_avg = epoch_loss / print_every
            print("%s (%d %d%%) %.4f" % (
                utils.time_since(start, epoch / n_epochs),
                epoch, epoch / n_epochs * 100, loss_avg))
    return LSTM


def eval(LSTM, valid_loader, batch_size, vocab, langs):
    """Evaluate an LSTM.

    Parameters
    ----------
    LSTM : LSTMClassifier
        A trained LSTMClassifier object.
    valid_loader : DataLoader
        A loader for the validation data.
    batch_size : int
    vocab : Vocab
        A vocabulary object for valid_loader data.
    langs : dict
        A dictionary mapping predictions to language names.

    """
    correct = 0
    n_examples = 0
    for data in valid_loader:
        if use_cuda:
            sentence = Variable(data["sentence"].cuda())
            label = Variable(data["language"].cuda())
        else:
            sentence = Variable(data["sentence"])
            label = Variable(data["language"])
        pred = LSTM(sentence)
        _, topi = pred.data.topk(1)
        sent_lists = torch.t(data["sentence"]).tolist()
        for batch_i, sent_list in enumerate(sent_lists):
            sent = (" ").join(utils.words_from_indices(vocab, sent_list))
            pred_lang_idx = int(topi[batch_i])
            true_lang_idx = int(label[batch_i])
            print("Sentence:", sent)
            print("Prediction:", langs[pred_lang_idx])
            print("Truth:", langs[true_lang_idx])
            n_examples += 1
            if pred_lang_idx == true_lang_idx:
                correct += 1
    print("Correctly predicted {correct} of {examples} [{percent}]%".format(
        correct=correct, examples=n_examples, percent=100*correct/n_examples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify English vs French")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bidirectional", action="store_true")
    FLAGS = parser.parse_args()
    langs = {0: "English", 1: "French"}

    VALID_SIZE = .2
    BATCH_SIZE = 1

    train_dset = dataset.sentences(FLAGS.data_dir)
    valid_dset = dataset.sentences(FLAGS.data_dir)

    train_sampler, valid_sampler = dataset.train_test_samplers(
        VALID_SIZE, len(train_dset))

    train_loader = dataset.padded_dataloader(
        train_dset, BATCH_SIZE, train_sampler, num_workers=4)
    valid_loader = dataset.padded_dataloader(
        valid_dset, BATCH_SIZE, valid_sampler, num_workers=4)

    print("Successfully loaded data.")

    if use_cuda:
        print("Found a CUDA-configured GPU.")
    else:
        print("Training on CPU.")
    LSTM = train(
        train_loader,
        BATCH_SIZE,
        train_dset.vocab.vocab_size,
        bidirectional=FLAGS.bidirectional)
    eval(LSTM, valid_loader, BATCH_SIZE, valid_dset.vocab, langs)

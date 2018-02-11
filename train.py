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


def train(train_loader, batch_size, vocab_size):
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
    for data in valid_loader:
        if use_cuda:
            sentence = Variable(data["sentence"].cuda())
            label = Variable(data["language"].cuda())
        else:
            sentence = Variable(data["sentence"])
            label = Variable(data["language"])
        pred = LSTM(sentence)
        sent_lists = data["sentence"].view(batch_size, -1).tolist()[0]
        print("Sentence:",
              (" ").join(utils.words_from_index(vocab, sent_lists)))
        _, topi = pred.data.topk(1)
        print("Prediction:", langs[int(topi[0])])
        print("Truth:", langs[int(label[0])])


if __name__ == "__main__":
    langs = {1: "English", 0: "French"}

    VALID_SIZE = .2
    BATCH_SIZE = 1

    data_dir = sys.argv[1]
    train_dset = dataset.sentences(data_dir)
    valid_dset = dataset.sentences(data_dir)

    train_sampler, valid_sampler = dataset.train_test_samplers(
        VALID_SIZE, len(train_dset))

    train_loader = dataset.padded_dataloader(
        train_dset, BATCH_SIZE, train_sampler, num_workers=4)
    valid_loader = dataset.padded_dataloader(
        valid_dset, BATCH_SIZE, valid_sampler, num_workers=4)

    print("Successfully loaded data.")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Found a CUDA-configured GPU.")
    else:
        print("Training on CPU.")
    LSTM = train(train_loader, BATCH_SIZE, train_dset.vocab.vocab_size)
    eval(LSTM, valid_loader, BATCH_SIZE, valid_dset.vocab, langs)


#!/usr/bin/env python
import unicodedata
import os
import re
import time
from math import floor


class Vocab:
    """The vocabulary associated with some data.

    Parameters
    ----------
    name : str
        The name of the encoded language.
        delims : True, optional
            To use an End of Sentence token and a Start of Sentence token.

    """

    def __init__(self, name, delims=False):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # count the words recorded
        if delims:
            self.add_word("SOS")
            self.add_word("EOS")

    @property
    def vocab_size(self):
        """The number of words in the vocabulary."""
        return self.n_words

    def add_word(self, word):
        """Add a word to the vocabulary."""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        """Add a sentence to the vocabulary."""
        for word in sentence.split(" "):
            self.add_word(word)


def unicode_to_ascii(s):
    """Convert from unicode encoding to ASCII."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn")


def normalize_string(s):
    """Remove certain special characters from a string."""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_langs(pair_file, pairs=False, reverse=False):
    """Prepare the input data for a language pairing file."""
    print("Reading lines...")
    lines = open(os.path.join(pair_file), encoding="utf-8").\
        read().strip().split("\n")
    if pairs:
        pairs = [[normalize_string(s) for s in li.split("\t")] for li in lines]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
        return pairs
    else:
        sents = [
            [(normalize_string(li.split("\t")[idx])) for li in lines]
            for idx in range(2)]
        if reverse:
            sents = list(reversed(sents))
        return sents


def indexes_from_sentence(vocab, sentence):
    """Convert an index to a sentence inside the given vocabulary."""
    return [vocab.word2index[word] for word in sentence.split(" ")]


def words_from_indexes(vocab, sentence_idx):
    """Convert some words to a list of indices inside the given vocabulary."""
    return [vocab.index2word[idx] for idx in sentence_idx]


def as_minutes(s):
    """Convert to minutes and return a string with minutes and seconds."""
    m = floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    """Return a formatted string timer."""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))

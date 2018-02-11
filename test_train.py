import unittest
import os
import dataset
import train
from utils import words_from_indices
from math import ceil
import torch


def setUpModule():
    print("Setting up...")
    global _data_dir
    _data_dir = "./data/fakefile.txt"
    global _text
    _text = ("This is English\tEs espanol.\n"
             "Today is a good day.\tHoy es buen dia.\n"
             "I have three sentences.\tTengo tres frases.").lower()
    global _words
    _words = ["PAD"]
    global _sentences
    _sentences = []
    for line in _text.split("\n"):
        for sent in line.split("\t"):
            _sentences.append(sent)
            for word in sent.split(" "):
                _words.append(word)
    print(set(_words))
    with open(_data_dir, "w") as f:
        f.write(_text)
    global langs
    langs = {0: "English", 1: "Spanish"}


def tearDownModule():
    print("Tearing down.")
    os.remove(_data_dir)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.VALID_SIZE = 1./3
        cls.BATCH_SIZE = 2

        cls.train_dset = dataset.sentences(_data_dir)
        cls.valid_dset = dataset.sentences(_data_dir)

        train_sampler, valid_sampler = dataset.train_test_samplers(
            cls.VALID_SIZE, len(cls.train_dset))

        cls.train_loader = dataset.padded_dataloader(
            cls.train_dset, cls.BATCH_SIZE, train_sampler, num_workers=1)
        cls.valid_loader = dataset.padded_dataloader(
            cls.valid_dset, cls.BATCH_SIZE, valid_sampler, num_workers=4)

    def testVocabUnity(self):
        self.assertEqual(self.train_dset.vocab.vocab_size,
                         self.valid_dset.vocab.vocab_size)

    def testVocabCounter(self):
        self.assertEqual(self.train_dset.vocab.vocab_size, len(set(_words)))

    def testTrainDataset(self):
        train_count = 0
        for elem in self.train_loader:
            train_count += 1
        num_train_sents = (1 - self.VALID_SIZE) * len(_sentences)
        # I haven't verified it will "ceil", but it should. The last batch
        # should be an incomplete one.
        num_train_batches = ceil(num_train_sents / self.BATCH_SIZE)
        self.assertEqual(train_count, num_train_batches)

    def testValidDataset(self):
        valid_count = 0
        for elem in self.valid_loader:
            valid_count += 1
        num_valid_sents = self.VALID_SIZE * len(_sentences)
        num_train_batches = ceil(num_valid_sents / self.BATCH_SIZE)
        self.assertEqual(valid_count, num_train_batches)


class TestLang(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.VALID_SIZE = 1./3
        cls.BATCH_SIZE = 2

        cls.train_dset = dataset.sentences(_data_dir)
        cls.valid_dset = dataset.sentences(_data_dir)

        train_sampler, valid_sampler = dataset.train_test_samplers(
            cls.VALID_SIZE, len(cls.train_dset))

        cls.train_loader = dataset.padded_dataloader(
            cls.train_dset, cls.BATCH_SIZE, train_sampler, num_workers=1)
        cls.valid_loader = dataset.padded_dataloader(
            cls.valid_dset, cls.BATCH_SIZE, valid_sampler, num_workers=4)

    def test_lang_order(self):
        eng_sents = [line.split("\t")[0] for line in _text.split("\n")]
        esp_sents = [line.split("\t")[1] for line in _text.split("\n")]

        for elem in self.valid_loader:
            print("SENT:")
            print(elem["sentence"])
            list_of_sentence_idx_lists = torch.t(elem["sentence"]).tolist()
            print("SENT LISTS:")
            print(list_of_sentence_idx_lists)
            one_sentence_idx_list = list_of_sentence_idx_lists[0]
            sent = (" ").join(words_from_indices(
                self.valid_dset.vocab, one_sentence_idx_list))
            lang_idx_of_sent_in_question = int(elem["language"][0])
            print("SENTENCE: ")
            print(sent)
            if langs[lang_idx_of_sent_in_question] == "english":
                self.assertTrue(sent in eng_sents)
            elif langs[lang_idx_of_sent_in_question] == "spanish":
                self.assertTrue(sent in esp_sents)


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.VALID_SIZE = 1./3
        cls.BATCH_SIZE = 2

        cls.train_dset = dataset.sentences(_data_dir)
        cls.valid_dset = dataset.sentences(_data_dir)

        train_sampler, valid_sampler = dataset.train_test_samplers(
            cls.VALID_SIZE, len(cls.train_dset))

        cls.train_loader = dataset.padded_dataloader(
            cls.train_dset, cls.BATCH_SIZE, train_sampler, num_workers=1)
        cls.valid_loader = dataset.padded_dataloader(
            cls.valid_dset, cls.BATCH_SIZE, valid_sampler, num_workers=4)

    def test_train(self):
        LSTM = train.train(
            self.train_loader,
            self.BATCH_SIZE,
            self.train_dset.vocab.vocab_size)

    def test_eval(self):
        LSTM = train.train(
            self.train_loader,
            self.BATCH_SIZE,
            self.train_dset.vocab.vocab_size)
        train.eval(
            LSTM,
            self.valid_loader,
            self.BATCH_SIZE,
            self.valid_dset.vocab,
            langs)


if __name__ == "__main__":
    unittest.main()

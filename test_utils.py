import unittest
import utils
import os


class TestReadLangs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._data_dir = "./data/fakefile.txt"
        # Must be all lowercase, must not end with \n
        cls._text = ("This is English\tEs espanol.\n"
                     "Today is a good day.\tHoy es buen dia.").lower()
        with open(cls._data_dir, "w") as f:
            f.write(cls._text)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._data_dir)

    def test_read_lang_pairs(self):
        pairs = utils.read_langs(self._data_dir, pairs=True, reverse=False)
        english = [li.split("\t")[0] for li in self._text.split("\n")]
        spanish = [li.split("\t")[1] for li in self._text.split("\n")]
        self.assertEqual(pairs, [[en, sp] for en, sp in zip(english, spanish)])

    def test_read_lang_pairs_reversed(self):
        pairs = utils.read_langs(self._data_dir, pairs=True, reverse=True)
        english = [li.split("\t")[0] for li in self._text.split("\n")]
        spanish = [li.split("\t")[1] for li in self._text.split("\n")]
        self.assertEqual(pairs, [[sp, en] for en, sp in zip(english, spanish)])

    def test_read_lang_sents(self):
        sents = utils.read_langs(self._data_dir, pairs=False, reverse=False)
        english = [li.split("\t")[0] for li in self._text.split("\n")]
        spanish = [li.split("\t")[1] for li in self._text.split("\n")]
        self.assertEqual(sents, [english, spanish])

    def test_read_lang_sents_reversed(self):
        sents = utils.read_langs(self._data_dir, pairs=False, reverse=True)
        english = [li.split("\t")[0] for li in self._text.split("\n")]
        spanish = [li.split("\t")[1] for li in self._text.split("\n")]
        self.assertEqual(sents, [spanish, english])

if __name__ == "__main__":
    unittest.main()

import torch
from torch.utils.data import Dataset, DataLoader
import utils
import numpy as np


class LangDataset(Dataset):
    """Create a dataset for a language or vocabulary.

    Paramters
    ---------
    data_dir:
        A directory to a .txt of paired vocabularies.
    transform:
        A Transform object.

    """

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        eng_sents, fra_sents = utils.read_langs(data_dir)
        self.input_data = eng_sents + fra_sents
        self.targets = np.concatenate(
            (np.zeros(len(eng_sents), dtype=np.int64),
             np.ones(len(fra_sents), dtype=np.int64)),
            0)
        self.vocab = utils.Vocab("all_data")
        for sent in self.input_data:
            self.vocab.add_sentence(sent)
        self.input_idxs = [utils.indexes_from_sentence(self.vocab, sent)
                           for sent in self.input_data]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        sentence = self.input_idxs[idx]
        target = self.targets[idx]
        sample = {"sentence": sentence, "language": target}
        if self.transform:
            sample = self.transform(sample)
        return sample


class toTensor(object):
    """A transformation for the LangDataset."""
    def __call__(self, sample):
        sample["sentence"] = torch.LongTensor(sample["sentence"])
        sample["label"] = torch.LongTensor([int(sample["language"])])
        return sample


def collate(samples):
    """The batching function for the LangDataset.
    
    Parameters
    ----------
    samples : list of dictionaries of the data.

    """

    out = pad_sequence([sample["sentence"] for sample in samples])
    return {"sentence": out,
            "language": torch.LongTensor([int(sample["language"])
                                          for sample in samples])}


def pad_sequence(sequences, batch_first=False):
    r"""Pad a list of variable length Variables with zero

    This function is borrowed from the source for 0.4.0 pre-release at
        http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html#pad_sequence
    with alterations from a well received Gist at
        https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and padds them to equal length. For example, if the input is list of
    sequences with size ``Lx*`` and if batch_first is False, and ``TxBx*``
    otherwise. The list of sequences should be sorted in the order of
    decreasing length.

    B is batch size. It's equal to the number of elements in ``sequences``.
    T is length longest sequence.
    L is length of the sequence.
    * is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size TxBx* or BxTx* where T is the
            length of longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Variable]): list of variable length sequences.
        batch_first (bool, optional): output will be in BxTx* if True, or in
            TxBx* otherwise

    Returns:
        Variable of size ``T x B x * `` if batch_first is False
        Variable of size ``B x T x * `` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]

    max_len = torch.LongTensor(
        [sequences[i].size(0) for i in range(len(sequences))]).max()
    trailing_dims = sequences[0].size()[1:]
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_variable = sequences[0].new(*out_dims).zero_()
    for i, variable in enumerate(sequences):
        length = variable.size(0)
        # use index notation to prevent duplicate references to the variable
        if batch_first:
            out_variable[i, :length, ...] = variable
        else:
            out_variable[:length, i, ...] = variable

    return out_variable


def sentences(path):
    return LangDataset(path, transform=toTensor())


def padded_dataloader(dset, batch_size, sampler, num_workers=4):
    """Provide dataloader where dset is padded to agree with next vec."""
    return DataLoader(
        dset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate)


def train_test_samplers(test_size, num_total, shuffle=True):
    """Inspired by the following Gist:
        https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
    """
    indices = list(range(num_total))
    split = int(np.floor(test_size * num_total))

    if shuffle is True:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    return (torch.utils.data.sampler.SubsetRandomSampler(train_idx),
            torch.utils.data.sampler.SubsetRandomSampler(valid_idx))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class LSTMClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 batch_size,
                 num_layers=1,
                 num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.num_layers = 1

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers)
        self.hidden2class = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hiddens = (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        if torch.cuda.is_available():
            return tuple([autograd.Variable(h.cuda()) for h in hiddens])
        else:
            return tuple([autograd.Variable(h) for h in hiddens])

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(sentence.size(0), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lbl_space = self.hidden2class(lstm_out[-1])
        lbl_score = F.log_softmax(lbl_space, dim=1)
        return lbl_score


class BiLSTMClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 batch_size,
                 num_layers=1,
                 num_classes=2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_classes = num_classes
        self.num_layers = num_layers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=self.num_layers,
            bidirectional=True)
        self.hidden2class = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hiddens = (
            torch.zeros(self.num_layers*2,
                        self.batch_size,
                        self.hidden_dim // 2),
            torch.zeros(self.num_layers*2,
                        self.batch_size,
                        self.hidden_dim // 2))
        if torch.cuda.is_available():
            return tuple([autograd.Variable(h.cuda()) for h in hiddens])
        else:
            return tuple([autograd.Variable(h) for h in hiddens])

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(sentence.size(0), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        lbl_space = self.hidden2class(lstm_out[-1])
        lbl_score = F.log_softmax(lbl_space, dim=1)
        return lbl_score



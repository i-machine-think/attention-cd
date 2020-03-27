import os
import torch

from collections import Counter


class UnknownDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            return dict.__getitem__(self, '<unk>')

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


class Dictionary(object):
    def __init__(self):
        self.word2idx = UnknownDict()
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        self.add_word('<unk>')

    def add_word(self, word, freq=1):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += freq
        self.total += freq
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab=None):
        self.dictionary = Dictionary()

        if vocab:
            self.dict_from_vocab(vocab)

        construct_dictionary = not vocab
        self.train = self.tokenize(os.path.join(path, 'train.txt'), construct_dictionary=construct_dictionary)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def dict_from_vocab(self, vocab):
        assert os.path.exists(vocab)

        with open(vocab, 'r') as f:
            for line in f:
                for word in line.split() + ['<eos>']:
                    self.dictionary.add_word(word)

        d = Dictionary()
        for wid, freq in sorted(self.dictionary.counter.items(), key=lambda x: x[1], reverse=True):
            d.add_word(self.dictionary.idx2word[wid], freq)
        self.dictionary = d


    def tokenize(self, path, construct_dictionary=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                if construct_dictionary:
                    for word in words:
                        self.dictionary.add_word(word)

        if construct_dictionary:
            d = Dictionary()
            for wid, freq in sorted(self.dictionary.counter.items(), key=lambda x: x[1], reverse=True):
                d.add_word(self.dictionary.idx2word[wid], freq)
            self.dictionary = d

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

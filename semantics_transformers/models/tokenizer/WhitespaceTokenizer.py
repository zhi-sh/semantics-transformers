# -*- coding: utf-8 -*-
# @DateTime :2021/3/2
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import collections, json, os
from typing import List, Iterable
from semantics_transformers.models.tokenizer import AbstractTokenizer


class WhitespaceTokenizer(AbstractTokenizer):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = [], do_lower_case: bool = False):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def get_vocab(self):
        return self.vocab

    def get_padding_idx(self):
        return self.word2idx.get(self.PAD, None)

    def tokenize(self, text: str) -> List[int]:
        if self.do_lower_case:
            text = text.lower()

        tokens_filtered = []
        tokens = text.split()
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
            else:
                if self.UNK in self.word2idx:
                    tokens_filtered.append(self.word2idx.get(self.UNK))
        return tokens_filtered

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'white-space-tokenizer.json'), 'w', encoding='utf-8') as fout:
            obj = {
                'vocab': list(self.word2idx.keys()),
                'stop_words': list(self.stop_words),
                'do_lower_case': self.do_lower_case
            }
            json.dump(obj, fout, indent=2, ensure_ascii=False)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'white-space-tokenizer.json'), encoding='utf-8') as fin:
            config = json.load(fin)
        return WhitespaceTokenizer(**config)

    @staticmethod
    def vocab_from_file(raw_text_file: str, max_vocab_size: int = 100000):
        counter = collections.Counter()

        with open(raw_text_file, encoding='utf-8') as fin:
            for line in fin:
                counter.update(line.strip().split())

        vocab = [WhitespaceTokenizer.PAD, WhitespaceTokenizer.UNK]
        pre_size = len(vocab)
        for w, c in counter.most_common(max_vocab_size - pre_size):
            vocab.append(w)

        return vocab

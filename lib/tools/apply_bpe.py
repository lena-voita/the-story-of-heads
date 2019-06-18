#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from collections import defaultdict


class BPEizer:
    def __init__(self, path, separator=' `'):
        """
        A tool that converts tokenized strings into BPE units given bpe rules
        Works by iteratively merging subword pairs with lowest priority, starting from individual characters
        :param path: path to a file with bpe merging rules. Either from subword_nmt or yandex internal bpe tool
            subword_nmt: file should start with #version: {some version} header and contain "{left_part right_part}" rules
            yandex internal: file shoud contain lines with "{left_part}\t{right_part}\t{priority}"
        :param separator: a string that will separates segments of a word;
            Note: subword_nmt's default separator is "@@ " (mind the space)

        Usage:
        >>> bpeizer = BPEizer(path='./data/ru.bpe.voc')
        >>> bpeizer.bpeize_token('транспонировали')
        'тран `сп `он `ир `овали'
        >>> bpeizer(['тридцать три треугольных матрицы транспонировали - транспонировали', ', да не вытранспонировали !'])
        ['тридцать три треуголь `ных мат `рицы тран `сп `он `ир `овали - тран `сп `он `ир `овали',
         ', да не выт `ран `сп `он `ир `овали !']
        """
        self.bpe_rules = defaultdict(lambda: float('inf'))
        self.separator = separator

        if self.is_yandex_bpe(path):
            self.mode = 'yandex'
            self.begin, self.end = '^$'
            for left, right, index in map(str.split, open(path)):
                self.bpe_rules[left, right] = int(index)

        elif self.is_rsenrich_bpe(path):
            self.mode = 'rsenrich'
            self.begin, self.end = '<w>', '</w>'
            f_rules = open(path)
            f_rules.readline()
            for i, (left, right) in enumerate(map(str.split, f_rules)):
                self.bpe_rules[left, right] = i
        else:
            raise NotImplementedError("BPE rules header is compatible with neither subword_nmt nor yandex bpe")

        self.escape_chars = {self.begin: chr(0x110000 - 2), self.end: chr(0x110000 - 1)}
        self.unescape_chars = {v: k for k, v in self.escape_chars.items()}

    def bpeize_token(self, chars: str):
        """ split a single token (str) into bpe units """
        tokens = [self.begin] + [self.escape_chars.get(c, c) for c in chars] + [self.end]
        if self.mode == 'rsenrich':
            last = tokens.pop()
            tokens[-1] += last  # automatically merge </w> with previous token

        while len(tokens) > 1:
            # find first bpe rule to match
            bpe_rule_priorities = [self.bpe_rules[prev, cur] for prev, cur in zip(tokens[:-1], tokens[1:])]

            chosen_ix = np.argmin(bpe_rule_priorities)
            if bpe_rule_priorities[chosen_ix] == float('inf'):
                break  # this is the end of the road, afro samurai!

            # apply it
            tokens = tokens[:chosen_ix] + [tokens[chosen_ix] + tokens[chosen_ix + 1]] + tokens[chosen_ix + 2:]

        assert tokens[0].startswith(self.begin) and tokens[-1].endswith(self.end)
        tokens[0] = tokens[0][len(self.begin):]
        tokens[-1] = tokens[-1][:-len(self.end)]
        tokens = [''.join([self.unescape_chars.get(c, c) for c in bpe])
                  for bpe in tokens if len(bpe) != 0]
        return self.separator.join(filter(len, tokens))

    def bpeize_line(self, line: str):
        """ converts a tokenized line into a bpe-ized line """
        return ' '.join(map(self.bpeize_token, line.split()))

    def __call__(self, text):
        if isinstance(text, (list, tuple)):
            return list(map(self, text))
        elif isinstance(text, str):
            return self.bpeize_line(text)
        else:
            raise ValueError("Expected string or list/tuple of strings but found {}".format(type(text)))

    @staticmethod
    def is_rsenrich_bpe(bpe_rules_path):
        """ Check if bpe rules were learned by https://github.com/rsennrich/subword-nmt """
        header = open(bpe_rules_path).readline()
        return header.startswith('#version:')

    @staticmethod
    def is_yandex_bpe(bpe_rules_path):
        """ Check if bpe rules were learned by internal Yandex tool """
        try:
            header = open(bpe_rules_path).readline()
            l, r, i = header.split('\t')  # check if this line contains 3 tabs
            return True
        except:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe_rules', required=True)
    args = parser.parse_args()

    bpeizer = BPEizer(args.bpe_rules)
    for l in sys.stdin:
        print(bpeizer.bpeize_line(l))

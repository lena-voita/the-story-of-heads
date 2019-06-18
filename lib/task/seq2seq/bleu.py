#!/usr/bin/env python3
# coding: utf-8

import argparse
from collections import Counter, namedtuple
import math
import os.path
import sys
import numpy as np

sys.path += [os.path.dirname(sys.argv[0])]

from .strutils import tokenize, al_num, all_chars__punct_tokens, all_chars__punct_tokens__foldcase
from .strutils import al_num__foldcase, chinese_tok, split_by_char_tok, equal_to_framework

SEED = 51
BleuResult = namedtuple('BleuResult', 'BLEU brevity_penalty ratio hyp_len ref_len BLEU_for_ngrams')


def best_match_length(references, cand, verbose=False):
    spl_cand_length = len(cand)
    diff = sys.maxsize
    for ref in references:
        spl_ref_length = len(ref)
        if not spl_ref_length:
            continue
        if spl_ref_length == spl_cand_length:
            return spl_ref_length
        elif abs(diff) == abs(spl_cand_length - spl_ref_length):
            diff = max(diff, spl_cand_length - spl_ref_length)
        elif abs(diff) > abs(spl_cand_length - spl_ref_length):
            diff = spl_cand_length - spl_ref_length
    best_len = max(spl_cand_length - diff, 0)
    if verbose and not best_len:
        print('WARNING: empty reference: ', repr((references, cand)), file=sys.stderr)
    return best_len


def brev_penalty(cand_length, best_match_length):
    if cand_length > best_match_length:
        return 1
    else:
        return math.exp(1 - float(best_match_length) / float(cand_length))


def split_into_ngrams(text, n):
    if n <= 0:
        raise ValueError('n should be a positive number!')
    return [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]


def compute_length_for_n(text, n_for_ngram):
    '''
    # split into words and count:
    # count - n
    '''
    unigram_count = len(text)
    if n_for_ngram > unigram_count:
        return 0
    else:
        return unigram_count - n_for_ngram + 1


def mod_precision_for_n(refs, cand, n, smoothed=False):
    cand_counter = Counter(split_into_ngrams(cand, n))
    ref_counters = [Counter(split_into_ngrams(ref, n)) for ref in refs]
    total_sum = 0
    for ngram, count_in_cand in cand_counter.items():
        max_count_in_refs = max(counter[ngram] for counter in ref_counters)
        total_sum += min(max_count_in_refs, count_in_cand)
    if smoothed and n > 1:
        return total_sum + 1, compute_length_for_n(cand, n) + 1
    return total_sum, compute_length_for_n(cand, n)


def logarithm(x):
    if x == 0:
        return -sys.maxsize - 1
    else:
        return math.log(x)


def print_summary(bleu_vals):
    bleu_mean, bleu_std = np.mean(bleu_vals), np.std(bleu_vals)
    summary_string = ("Mean BLEU: %.4f; 95%% CI: [%.4f, %.4f]; std=%.4f" %
        (bleu_mean, bleu_mean - 1.96 * bleu_std, bleu_mean + 1.96 * bleu_std, bleu_std))
    print(summary_string)


class Bleu(object):
    def __init__(self, normalize_func=None, smoothed=False, cached=False, language=None, verbose=False):
        self.cand_len = 0
        self.best_ref_len = 0
        self.brevity_penalty = 0
        self.mod_precision = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.normalize_func = normalize_func
        self.smoothed = smoothed
        self.cached = cached
        self.language = language
        if cached:
            self.cand_len_vals = []
            self.best_ref_len_vals = []
            self.mod_precision_vals = []
        self.verbose = verbose

    def process_next(self, cand, refs, **kwargs):
        if self.normalize_func is not None:
            cand = tokenize(self.normalize_func(cand, self.language))
            refs = [tokenize(self.normalize_func(ref, self.language)) for ref in refs]
        else:
            cand = tokenize(cand)
            refs = [tokenize(ref) for ref in refs]
        self.last__cand_len = compute_length_for_n(cand, 1)
        self.cand_len += self.last__cand_len
        self.last__best_ref_len = best_match_length(refs, cand, verbose=self.verbose)
        self.best_ref_len += self.last__best_ref_len
        self.last_mp = []
        for i in range(4):
            self.last_mp.append(mod_precision_for_n(refs, cand, i + 1, smoothed=self.smoothed))
            self.mod_precision[i][0] += self.last_mp[i][0]
            self.mod_precision[i][1] += self.last_mp[i][1]

        if self.cached:
            self.cand_len_vals.append(self.last__cand_len)
            self.best_ref_len_vals.append(self.last__best_ref_len)
            self.mod_precision_vals.append(self.last_mp)

    def _compute_bleu(self, cand_len, best_ref_len, mod_precision, sentence_level=False):
        brevity_penalty = brev_penalty(cand_len, best_ref_len)
        bleu_for_ngram = [0, 0, 0, 0]
        for i in range(4):
            if mod_precision[i][0] > 0.0 and mod_precision[i][1] > 0.0 :
                bleu_for_ngram[i] = round(float(mod_precision[i][0]) / float(mod_precision[i][1]), 4)
            else:
                bleu_for_ngram[i] = 0.0
        average = 0
        for i in range(4):
            if sentence_level:
                nonzero = mod_precision[i][1] > 0.0
            else:
                nonzero = mod_precision[i][0] > 0.0 and mod_precision[i][1] > 0.0
                if not nonzero:
                    average += 0.25 * (-sys.maxsize)
            if nonzero:
                average += 0.25 * logarithm(float(mod_precision[i][0]) / float(mod_precision[i][1]))
        total_bleu = round(brevity_penalty * math.exp(average), 4)
        return BleuResult(total_bleu, brevity_penalty, round(float(cand_len) / float(best_ref_len), 4), cand_len, best_ref_len, bleu_for_ngram)

    def result_for_last(self):
        return self._compute_bleu(self.last__cand_len, self.last__best_ref_len, self.last_mp, True)

    def total(self):
        return self._compute_bleu(self.cand_len, self.best_ref_len, self.mod_precision)

    def bootstrap_sample(self, n_times=1000, seed=None):
        rng = np.random.RandomState(seed)
        if not self.cached:
            return None
        bleu_vals = []
        for i in range(n_times):
            inds = rng.randint(0, len(self.cand_len_vals), len(self.cand_len_vals))
            cand_len = sum([self.cand_len_vals[i] for i in inds])
            best_ref_len = sum([self.best_ref_len_vals[i] for i in inds])
            mod_precision = sum([np.array(self.mod_precision_vals[i]) for i in inds])
            bleu_vals.append(self._compute_bleu(cand_len, best_ref_len, mod_precision)[0])
        return np.array(bleu_vals)


if __name__ == '__main__':
    t_options = {'simple': al_num__foldcase,
                 'case-sensitive': al_num,
                 'punctuation': all_chars__punct_tokens__foldcase,
                 'c-s-punctuation': all_chars__punct_tokens,
                 'ch': chinese_tok,
                 'split-by-char': split_by_char_tok,
                 'framework': equal_to_framework}
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenization', help='''Tokenization options:
    default - split text by spaces
    simple = alphanumerics only,
    case-sensitive = with small letters,
    punctuation = with punctuation marks as separate tokens,
    c-s-punctuation = case-sensitive + punctuation,
    split-by-char = set space between all characters,
    framework = lang-specific replacements + unicode category tokenization''',
    choices=t_options.keys())
    parser.add_argument('-c', '--candidate', type=int, nargs='+', help='Hypothesis column number.', required=True)
    parser.add_argument('-r', '--reference', help='Reference column number (range or int)')
    parser.add_argument('--all', help='Bleu scores for all queries.', action='store_true')
    parser.add_argument('-s', '--smoothed', action='store_true', default=False, help='Use to compute smoothed BLEU')
    parser.add_argument('-l', '--language', help='Dst-side language')
    parser.add_argument('--bootstrap-sampling-n', type=int,
                        help='Run bootstrap sampling n times for BLEU CI estimate.', default=0)
    parser.add_argument('--compare', help='Compare Bleu scores for two MT systems', action='store_true')
    args = parser.parse_args()

    if args.compare and len(args.candidate) != 2:
        raise AssertionError('It should specify 2 hypothesis columns if `--compare` flag used')
    if args.compare and args.all:
        raise AssertionError('Could not evaluate BLEU score for each query if `--compare` flag used')

    if ':' in args.reference:
        r_start, r_end = args.reference.split(':')
        reference = slice(int(r_start), int(r_end) if len(r_end) > 0 else None)
    else:
        reference = int(args.reference)

    bleu_opts = {
        'normalize_func': t_options[args.tokenization] if args.tokenization else None,
        'smoothed': args.smoothed,
        'cached': bool(args.bootstrap_sampling_n) or args.compare,
        'language': args.language,
        }

    b_first = Bleu(verbose=True, **bleu_opts)
    if args.compare:
        b_second = Bleu(verbose=True, **bleu_opts)

    for i, line in enumerate(sys.stdin):  # for candidate and set of references in corpus compute process_next
        line = line.rstrip('\n')
        if not line:
            continue
        text_data = line.rstrip().split('\t')
        refs = [text_data[reference]] if isinstance(reference, int) else text_data[reference]
        refs = [ref for ref in refs if ref]
        if not refs:
            print('Error: no data found in {} column, line {}'.format(args.reference, i + 1), file=sys.stderr)
            continue
        if len(text_data) < 2:
            text_data += ['']
        cand_first = text_data[args.candidate[0]]
        b_first.process_next(cand_first, refs)
        if args.compare:
            cand_second = text_data[args.candidate[1]]
            b_second.process_next(cand_second, refs)

        #if not cand:
        #    print >> sys.stderr, 'Error: no data found in %d column, line %i' % (args.r, i + 1)
        #    sys.exit(1)
        if args.all:
            print(i, b_first.result_for_last()[0])

    if not args.compare:
        print(b_first.total())
        if args.bootstrap_sampling_n:
            bleu_vals = b_first.bootstrap_sample(args.bootstrap_sampling_n, seed=SEED)
            print_summary(bleu_vals)
    else:
        sampling_n = args.bootstrap_sampling_n if args.bootstrap_sampling_n > 0 else 1000

        print("---\nFirst system stats:" )
        print(b_first.total())
        bleu_vals_first = b_first.bootstrap_sample(sampling_n, seed=SEED)
        print_summary(bleu_vals_first)

        print("---\nSecond system stats:" )
        print(b_second.total())
        bleu_vals_second = b_second.bootstrap_sample(sampling_n, seed=SEED)
        print_summary(bleu_vals_second)

        delta = bleu_vals_first - bleu_vals_second
        bootstrap_p_value = np.mean(delta > 0)
        print("---\nSystem %d is better. Significance test results:" % (1 if bootstrap_p_value > 0.5 else 2))
        print("Paired boostrap p-value = %.3f" % min(bootstrap_p_value, 1 - bootstrap_p_value))



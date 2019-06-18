# coding: utf-8

from codecs import iterdecode
import re
import sys
import unicodedata


def normalize_table_lang(text, lang=None):
    """According to normalization done in framework"""
    if lang == 'ru':
        # replace capital and small letters IO -> IE
        return text.replace(u'\u0401', u'\u0415').replace(u'\u0451', u'\u0435')
    elif lang == 'ro':
        # replace capital and small letters S and T with cedilla -> comma below
        return text.replace(u'\u015F', u'\u0219').replace(u'\u015E',
            u'\u0218').replace(u'\u0163', u'\u021b').replace(u'\u0162', u'\u021a')
    elif lang == 'tr':
        # replace capital and small letters with circumflex
        return text.replace(u'\u00C2', u'\u0041').replace(u'\u00E2',
            u'\u0061').replace(u'\u00CE', u'\u0049').replace(u'\u00EE',
            u'\u0069').replace(u'\u00DB', u'\u0055').replace(u'\u00FB', u'\u0075')
    else:
        return text


def unicode_category_tokenize(text, lang=None):
    import regex
    re_for_split = regex.compile(
            u'(?u)[\p{Punctuation}\p{Separator}\p{Other}\p{Sm}\p{So}\p{Sc}]+')
    return u' '.join(tok for tok in re_for_split.split(text) if tok)


def chinese_tok(text, lang=None):
    """ставит между всеми символами пробелы"""
    # '''from meteor_ext import make_tmp_file, get_random_filename
    # import os
    #
    # tmpfile = make_tmp_file(pre=get_random_filename())
    # tmpfile.write(text.encode('utf-8'))
    # args = ['/place/framework/metrics/stanford-segmenter/segment.sh', 'ctb', tmpfile.name, encoding, '0']
    # p = Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # out_data, err_data = p.communicate()
    # tmpfile.close()
    # os.unlink(tmpfile.name)
    # return out_data'''
    # from itertools import cycle
    return ' '.join(text)

def split_by_char_tok(text, lang=None):
    return ' '.join(text)

def tokenize(text):
    return text.split()

def lower(text, lang=None):
    return text.lower()

def upper(text, lang=None):
    return text.upper()

def foldcase(text, lang=None):
    """приводит текст к одному регистру (верхнему)"""
    # folds case according to language
    # TODO: set locale by lang so that some letters are folded correctly
    # (e.g. turkish i without dot)
    return upper(text)

def join_tokens(text):
    return u' '.join(tokenize(text))


def separate_punctuation(text, lang=None):
    """отделяет пунктуацию и символы (Po/So/Ps/Pe/Sc/-) двумя пробелами, после ' ставит один пробел"""
    new_chars = []
    for character in text:
        if character == u"'":
            new_chars.append(character + u' ')
        elif unicodedata.category(character) in ('Po', 'So', 'Ps', 'Pe', 'Sc')\
         or character == u"-":
            new_chars.append(u' ' + character + u' ')
        else:
            new_chars.append(character)
    return "".join(new_chars)


def alphanum(text, lang=None):
    """заменяет все не-alphanumeric (\W, Unicode) на пробел"""
    #TODO: do not remove currency signs
    non_alphanum = re.compile(u'\W', re.UNICODE)
    text = non_alphanum.sub(' ', text)
    return text


def func_chain(*funcs):
    """Returns a function that chains parameter functions"""
    def result_func(text, lang=None):
        result = text
        for func in funcs:
            result = func(result, lang)
        return result
    return result_func


def normalize_space(u_text, lang=None):
    """стирает пробельные символы, заменяя их на один пробел"""
    return ' '.join(u_text.split())

def xlines(fileobj, encoding='utf_8_sig', keepends=False):
    for line in iterdecode(fileobj, encoding):
        if not keepends:
            line = line.rstrip('\r\n')
        yield line

# only alphanumeric characters are kept
al_num = func_chain(alphanum, normalize_space)
# only alphanumeric characters are kept, the rest is case-folded
al_num__foldcase = func_chain(foldcase, alphanum, normalize_space)
all_chars__as_is = func_chain()
# all characters are folded in case
all_chars__foldcase = func_chain(foldcase, normalize_space)
# punctuation becomes separate tokens
all_chars__punct_tokens = func_chain(separate_punctuation, normalize_space)
# punctuation becomes separate tokens, all characters are folded in case
all_chars__punct_tokens__foldcase = func_chain(foldcase, separate_punctuation, normalize_space)
# as is in eval framework
equal_to_framework = func_chain(normalize_table_lang, foldcase, unicode_category_tokenize)

if __name__ == '__main__':
    funcs = {'-s': al_num__foldcase, '-p': all_chars__punct_tokens__foldcase,
             '-cs': al_num, '-csp': all_chars__punct_tokens}
    a = sys.argv[1]
    for line in xlines(sys.stdin):
        print(u''.join(map(funcs[a], line.split('\t'))))
import collections
import sys


class BaseVoc:
    @property
    def bos(self):
        raise NotImplementedError()

    @property
    def eos(self):
        raise NotImplementedError()

    def ids(self, words):
        raise NotImplementedError()

    def words(self, ids):
        raise NotImplementedError()

    def size(self):
        raise NotImplementedError()


class Voc:
    @property
    def bos(self):
        return 0

    @property
    def eos(self):
        return 1

    @property
    def _unk(self):
        return 2

    def ids(self, words):
        if isinstance(words, (list, tuple)):
            return [self.ids(word) for word in words]
        return self._voc.get(words, self._unk)

    def words(self, ids):
        if isinstance(ids, (list, tuple)):
            return [self.words(id) for id in ids]
        return self._ivoc[ids]

    def size(self):
        return self._size

    @staticmethod
    def compile(corpus_filename, max_words, index=0):
        # Accumulate frequencies.
        freqs = collections.defaultdict(int)
        with open(corpus_filename) as corpus:
            for line in corpus:
                line = line.strip('\n')
                if not line:
                    continue
                for word in line.split(' '):
                    freqs[word.split('|||')[index]] += 1

        # Sort by frequency.
        freq_and_word = lambda item: item[::-1]
        most_frequent = sorted(freqs.items(), key=freq_and_word, reverse=True)

        # Create voc.
        obj = Voc()
        voc = { '_BOS_': obj.bos, '_EOS_': obj.eos }
        id = 3
        total_covered_freq = 0
        for word, freq in most_frequent[:max_words]:
            voc[word] = id
            id += 1
            total_covered_freq += freq

        # Report coverage.
        total_freq = sum(freqs.values())
        msg = 'Voc %r: %i words, %.3f%% coverage' % (
            corpus_filename,
            id,
            total_covered_freq * 100 / total_freq,
            )
        print(msg, file=sys.stderr, flush=True)

        # Return.
        obj.__setstate__((voc,))
        return obj

    def __getstate__(self):
        return self._voc,

    def __setstate__(self, state):
        # Load direct vocabulary.
        self._voc, = state

        # Fill inverse vocabulary.
        self._ivoc = {}
        for k, v in self._voc.items():
            self._ivoc[v] = k
        self._ivoc[self.bos] = '_BOS_'
        self._ivoc[self.eos] = '_EOS_'
        self._ivoc[self._unk] = '_UNK_'

        # Compute size
        self._size = max(self._voc.values()) + 1

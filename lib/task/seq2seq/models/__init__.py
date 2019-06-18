from ..inference import translate_lines
from lib.task.seq2seq.inference import TranslateModel, GreedyDecoder, PenalizedBeamSearchDecoder
from ..data import make_batch_data, make_batch_placeholder
from functools import lru_cache
from itertools import chain, islice


class ModelBase:
    def encode_decode(self, batch, is_train):
        """ Encode input sequence and decode rdo for output sequence """
        raise NotImplementedError()

    def _get_batch_sample(self):
        return [("i saw a cat", "i write the code")]

    def make_feed_dict(self, batch, **kwargs):
        batch_data = make_batch_data(batch, self.inp_voc, self.out_voc, force_bos=self.hp.get('force_bos', True), **kwargs)
        return batch_data


class TranslateModelBase(TranslateModel, ModelBase):
    """
       A base class that most seq2seq models depend on.
       Must have following fields: name, inp_voc, out_voc, loss
    """
    def translate_lines(self, lines, ingraph=True, ingraph_mode='beam_search',
                        unbpe=True, batch_size=None, dumper=None, **flags):
        """ Translate multiple lines with the model """
        if ingraph:
            translator = self.get_ingraph_translator(mode=ingraph_mode, back_prop=False, **flags)
        else:
            translator = self.get_translator(**flags)

        replace_unk = flags.get('replace', self.hp.get('replace', False))

        if batch_size is None:
            lines_batched = [lines]
        else:
            lines = iter(lines)
            lines_batched = list(iter(lambda: tuple(islice(lines, batch_size)), ()))

        outputs = (translate_lines(batch_lines, translator, self, self.out_voc, replace_unk, unbpe, dumper=dumper)
                   for batch_lines in lines_batched)

        return list(chain(*outputs))

    def predict(self):
        self.get_predictor().main()

    @lru_cache()
    def get_ingraph_translator(self, mode='beam_search', **flags):
        """
        Creates a symbolic translation graph on a batch of placeholders.
        Used to translate numeric data.
        :param mode: 'greedy', 'sample', or 'beam_search'
        :param flags: anything else you want to pass to decoder, encode, decode, sample, etc.
        :return: a class with .best_out, .best_scores containing symbolic tensors for translations
        """
        batch_data_sample = self.make_feed_dict(self._get_batch_sample())
        batch_placeholder = make_batch_placeholder(batch_data_sample)
        return self.symbolic_translate(batch_placeholder, mode, **flags)

    def symbolic_translate(self, batch_placeholder, mode='beam_search', **flags):
        """
        A function that takes a dict of symbolic inputs and outputs symolic translations
        :param batch_placeholder: a dict of symbolic inputs {'inp':int32[batch, time]}
        :param mode: str: 'greedy', 'sample', 'beam_search' or a decoder class
        :param flags: anything else you want to pass to decoder, encode, decode, sample, etc.
        :return: a class with .best_out, .best_scores containing symbolic tensors for translations
        """
        flags = dict(self.hp, **flags)

        if mode in ('greedy', 'sample'):
            flags['sampling_strategy'] = 'random' if mode == 'sample' else 'greedy'
            return GreedyDecoder(
                model=self.get_translate_model(),
                batch_placeholder=batch_placeholder,
                **flags
            )
        elif mode == 'beam_search':
            return PenalizedBeamSearchDecoder(
                model=self.get_translate_model(),
                batch_placeholder=batch_placeholder,
                **flags
            )
        elif callable(mode):
            return mode(self.get_translate_model(), batch_placeholder, **flags)
        else:
            raise ValueError("Invalid mode : %s" % mode)

    def get_translate_model(self):
        if hasattr(self, 'translate_model'):
            return self.translate_model

        return self
